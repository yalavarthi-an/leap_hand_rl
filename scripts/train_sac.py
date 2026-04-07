"""
train_sac.py — Soft Actor-Critic for MuJoCo Playground (MJX).

A complete JAX/Flax implementation of SAC adapted for GPU-parallel
MJX environments, specifically LeapCubeReorient.

Key design decisions:
  - All networks are Flax Linen modules (JAX-native neural networks)
  - Replay buffer stored as JAX arrays on GPU (fast sampling)
  - Environment stepped via jax.vmap for parallel data collection
  - Automatic entropy temperature (alpha) tuning
  - Twin Q-networks to prevent overestimation

Usage:
  cd ~/leap_hand_rl/mujoco_playground
  uv --no-config run python ../scripts/train_sac.py --seed 0

Author: Anish Yalavarthi (CS 5180 Project)
"""

import os
import time
import json
import argparse
from datetime import datetime
from functools import partial

# --- JAX ecosystem ---
# jax: core library (jit, vmap, grad)
# jax.numpy: GPU-accelerated NumPy replacement
# flax.linen: neural network library for JAX (like PyTorch's nn.Module)
# optax: gradient-based optimizers for JAX (Adam, SGD, etc.)
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training.train_state import TrainState

# --- MuJoCo Playground ---
from mujoco_playground import registry

# =========================================================================
# Configuration
# =========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="SAC for LeapCubeReorient")

    # Experiment
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--log_dir", type=str, default="./logs")

    # Environment
    p.add_argument("--env_name", type=str, default="LeapCubeReorient")
    p.add_argument("--num_envs", type=int, default=256,
                   help="Parallel envs for data collection (less than PPO — "
                        "SAC relies on replay buffer, not massive parallelism)")
    p.add_argument("--episode_length", type=int, default=1000)

    # SAC hyperparameters
    p.add_argument("--total_timesteps", type=int, default=50_000_000)
    p.add_argument("--learning_rate", type=float, default=3e-4,
                   help="Adam LR for actor, critic, and alpha")
    p.add_argument("--buffer_size", type=int, default=1_000_000,
                   help="Replay buffer capacity. 1M transitions ≈ 138 MB on GPU")
    p.add_argument("--batch_size", type=int, default=256,
                   help="Minibatch size sampled from replay buffer each update")
    p.add_argument("--gamma", type=float, default=0.99,
                   help="Discount factor — same as PPO baseline for fair comparison")
    p.add_argument("--tau", type=float, default=0.005,
                   help="Soft update coefficient: target = (1-τ)×target + τ×current")
    p.add_argument("--learning_starts", type=int, default=25000,
                   help="Random exploration steps before first gradient update. "
                        "Fills buffer with diverse initial experiences.")
    p.add_argument("--policy_frequency", type=int, default=2,
                   help="Update actor every N critic updates (delayed policy update)")
    p.add_argument("--target_entropy_scale", type=float, default=1.0,
                   help="Multiplier for target entropy. 1.0 = standard -dim(A)")

    # Network architecture
    p.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 256],
                   help="Hidden layer sizes for actor and critic networks")

    # Logging
    p.add_argument("--log_every", type=int, default=5000,
                   help="Print metrics every N env steps")
    p.add_argument("--eval_every", type=int, default=500_000,
                   help="Run eval episodes every N env steps")
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument("--save_every", type=int, default=2_000_000)
    p.add_argument("--use_tb", action="store_true", help="Enable TensorBoard logging")

    args = p.parse_args()
    if args.run_name is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"sac_{args.env_name}_s{args.seed}_{ts}"
    return args


# =========================================================================
# Step 2: Actor Network (Gaussian Policy with Tanh Squashing)
# =========================================================================

class Actor(nn.Module):
    """
    Stochastic policy: obs → mean, log_std → sample action via reparameterization.

    Architecture:
      obs (57) → Dense(256) → ReLU → Dense(256) → ReLU → mean (16), log_std (16)
      action = tanh(mean + exp(log_std) × noise)

    The tanh squashing bounds actions to [-1, 1], which gets scaled to
    the environment's action range. The log_prob correction accounts for
    the change of variables from Gaussian to tanh-squashed distribution.
    """
    action_dim: int
    hidden_dims: tuple = (256, 256)
    LOG_STD_MIN: float = -20.0  # Prevents std from collapsing to 0
    LOG_STD_MAX: float = 2.0    # Prevents std from exploding

    @nn.compact
    def __call__(self, obs):
        # Shared hidden layers
        x = obs
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)

        # Two output heads: mean and log_std
        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)

        # Clamp log_std to prevent numerical issues
        # Without this, std could become 0 (deterministic, no gradients)
        # or infinity (pure noise, no learning)
        log_std = jnp.clip(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

        return mean, log_std


def sample_action(actor_params, actor, obs, key):
    """
    Sample action using the reparameterization trick.

    Instead of sampling directly from the policy (which blocks gradients),
    we sample noise ~ N(0,1) and transform it:
      action = tanh(mean + std × noise)

    This lets gradients flow through the sampling process because the
    randomness (noise) is external to the computation graph.

    Also computes log_prob with the tanh correction:
      log π(a|s) = log N(u|mean, std) - Σ log(1 - tanh²(u))

    The correction term accounts for the Jacobian of the tanh transformation.
    Without it, the entropy estimates would be wrong and α tuning would fail.
    """
    mean, log_std = actor.apply(actor_params, obs)
    std = jnp.exp(log_std)

    # Reparameterization trick
    noise = jax.random.normal(key, mean.shape)
    u = mean + std * noise          # Pre-squash action (Gaussian)
    action = jnp.tanh(u)            # Squashed action in [-1, 1]

    # Log probability with tanh correction
    # Step 1: log prob under Gaussian
    log_prob = -0.5 * (((u - mean) / (std + 1e-8)) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi))
    log_prob = log_prob.sum(axis=-1)  # Sum across action dimensions

    # Step 2: Subtract log-determinant of tanh Jacobian
    # This is the "squashing correction" — mathematically derived from
    # the change of variables formula for probability distributions
    log_prob -= jnp.sum(jnp.log(1 - action ** 2 + 1e-6), axis=-1)

    return action, log_prob


# =========================================================================
# Step 3: Twin Critic Networks (Q-functions)
# =========================================================================

class TwinCritic(nn.Module):
    """
    Twin Q-networks: (obs, action) → Q₁, Q₂

    Both Q-networks share the same architecture but have INDEPENDENT weights.
    They're defined in one module for convenience, but trained independently.

    Input: obs (57) concatenated with action (16) = 73 dimensions
    Output: Two scalar Q-values

    Architecture per Q-network:
      (obs, action) (73) → Dense(256) → ReLU → Dense(256) → ReLU → Q-value (1)
    """
    hidden_dims: tuple = (256, 256)

    @nn.compact
    def __call__(self, obs, action):
        # Concatenate state and action — this is what makes it Q(s,a)
        # not V(s) like PPO's critic
        x = jnp.concatenate([obs, action], axis=-1)

        # Q-network 1
        q1 = x
        for dim in self.hidden_dims:
            q1 = nn.Dense(dim)(q1)
            q1 = nn.relu(q1)
        q1 = nn.Dense(1)(q1).squeeze(-1)  # (batch,) not (batch, 1)

        # Q-network 2 (independent weights due to separate Dense layers)
        q2 = x
        for dim in self.hidden_dims:
            q2 = nn.Dense(dim)(q2)
            q2 = nn.relu(q2)
        q2 = nn.Dense(1)(q2).squeeze(-1)

        return q1, q2


# =========================================================================
# Step 4: Replay Buffer (GPU-Resident)
# =========================================================================

class ReplayBuffer:
    """
    Fixed-size circular buffer stored as JAX arrays on GPU.

    Unlike PPO which discards data after each update, SAC stores ALL
    transitions and samples random minibatches for learning. This is
    the key to SAC's sample efficiency.

    Memory layout (for 1M capacity, LeapCubeReorient):
      obs:      (1_000_000, 57)  ← 57 MB
      actions:  (1_000_000, 16)  ← 16 MB
      rewards:  (1_000_000,)     ← 4 MB
      next_obs: (1_000_000, 57)  ← 57 MB
      dones:    (1_000_000,)     ← 4 MB
      Total:    ~138 MB on GPU

    The buffer uses a circular pointer — when full, new data overwrites
    the oldest transitions. This means the buffer always contains the
    most recent 1M transitions.
    """

    def __init__(self, obs_dim: int, action_dim: int, capacity: int):
        self.capacity = capacity
        self.obs = jnp.zeros((capacity, obs_dim))
        self.actions = jnp.zeros((capacity, action_dim))
        self.rewards = jnp.zeros((capacity,))
        self.next_obs = jnp.zeros((capacity, obs_dim))
        self.dones = jnp.zeros((capacity,))
        self.size = 0       # How many valid transitions are stored
        self.ptr = 0        # Where to write next

    def add_batch(self, obs, actions, rewards, next_obs, dones):
        """
        Add a batch of transitions from parallel environments.

        All inputs are JAX arrays. obs shape: (num_envs, obs_dim), etc.
        The batch is written starting at self.ptr, wrapping around if needed.
        """
        batch_size = obs.shape[0]
        indices = (jnp.arange(batch_size) + self.ptr) % self.capacity

        self.obs = self.obs.at[indices].set(obs)
        self.actions = self.actions.at[indices].set(actions)
        self.rewards = self.rewards.at[indices].set(rewards)
        self.next_obs = self.next_obs.at[indices].set(next_obs)
        self.dones = self.dones.at[indices].set(dones)

        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, key: jax.Array, batch_size: int):
        """
        Sample a random minibatch of transitions.

        This is the core of off-policy learning — we sample uniformly
        from ALL stored transitions, regardless of when they were collected
        or which version of the policy generated them.
        """
        indices = jax.random.randint(key, (batch_size,), 0, self.size)
        return (
            self.obs[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_obs[indices],
            self.dones[indices],
        )


# =========================================================================
# Step 5: SAC Update Functions (The Core Algorithm)
# =========================================================================

@jax.jit
def update_critic(
    critic_state,       # TrainState for twin critics
    target_critic_params,  # Frozen target network parameters
    actor_state,        # Current actor (for sampling next actions)
    log_alpha,          # Current entropy temperature (log scale)
    batch,              # (obs, actions, rewards, next_obs, dones)
    gamma,              # Discount factor
    key,                # PRNG key for sampling next actions
):
    """
    Update twin Q-networks toward the TD target.

    The target is:
      y = r + γ × (1-done) × (min(Q₁_target, Q₂_target) - α × log π(a'|s'))

    where a' is sampled from the CURRENT policy (not the one that collected the data).
    """
    obs, actions, rewards, next_obs, dones = batch
    key, action_key = jax.random.split(key)

    # Sample next actions from CURRENT policy
    next_actions, next_log_probs = sample_action(
        actor_state.params, actor_state.apply_fn, next_obs, action_key
    )

    # Compute target Q-values using TARGET networks (frozen, stable)
    alpha = jnp.exp(log_alpha)
    next_q1, next_q2 = critic_state.apply_fn.apply(
        target_critic_params, next_obs, next_actions
    )
    next_q = jnp.minimum(next_q1, next_q2)  # Pessimistic estimate
    next_q = next_q - alpha * next_log_probs  # Entropy bonus

    # TD target: r + γ(1-done)(Q_target - α log π)
    target_q = rewards + gamma * (1.0 - dones) * next_q

    # Critic loss: MSE between predicted Q and target
    def critic_loss_fn(critic_params):
        q1, q2 = critic_state.apply_fn.apply(critic_params, obs, actions)
        loss = 0.5 * (jnp.mean((q1 - target_q) ** 2) +
                       jnp.mean((q2 - target_q) ** 2))
        return loss, {"q1_mean": q1.mean(), "q2_mean": q2.mean(),
                       "critic_loss": loss}

    (loss, info), grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(
        critic_state.params
    )
    critic_state = critic_state.apply_gradients(grads=grads)

    return critic_state, info, key


@jax.jit
def update_actor_and_alpha(
    actor_state,        # TrainState for actor
    critic_state,       # Current critic (for evaluating new actions)
    log_alpha,          # Current log(α) — we optimize log(α) not α for stability
    alpha_optimizer,    # Optax optimizer state for alpha
    obs,                # Batch of observations
    target_entropy,     # -dim(A) = -16
    key,                # PRNG key
):
    """
    Update the actor to maximize Q-value while maintaining entropy.

    Actor loss: E[ α × log π(a|s) - min(Q₁, Q₂) ]
                    ^^^^^^^^^^^^^^   ^^^^^^^^^^^^^
                    "stay random"    "get high Q"

    We MINIMIZE this, so:
      - Minimizing α×log_prob = maximizing entropy (more random)
      - Minimizing -Q = maximizing Q (better actions)
    """
    key, action_key = jax.random.split(key)
    alpha = jnp.exp(log_alpha)

    # Actor loss
    def actor_loss_fn(actor_params):
        actions, log_probs = sample_action(
            actor_params, actor_state.apply_fn, obs, action_key
        )
        q1, q2 = critic_state.apply_fn.apply(critic_state.params, obs, actions)
        q_min = jnp.minimum(q1, q2)

        # loss = E[α log π - Q]
        loss = jnp.mean(alpha * log_probs - q_min)
        return loss, {"actor_loss": loss, "mean_log_prob": log_probs.mean()}

    (actor_loss, actor_info), actor_grads = jax.value_and_grad(
        actor_loss_fn, has_aux=True
    )(actor_state.params)
    actor_state = actor_state.apply_gradients(grads=actor_grads)

    # Alpha (entropy temperature) loss
    # If policy is too deterministic (log_prob > target_entropy):
    #   → alpha_loss is positive → gradient pushes α up → more exploration
    # If policy is too random (log_prob < target_entropy):
    #   → alpha_loss is negative → gradient pushes α down → less exploration
    def alpha_loss_fn(log_alpha_val):
        return -jnp.mean(
            log_alpha_val * (actor_info["mean_log_prob"] + target_entropy)
        )

    alpha_loss, alpha_grad = jax.value_and_grad(alpha_loss_fn)(log_alpha)
    alpha_updates, alpha_optimizer = optax.adam(3e-4).update(
        alpha_grad, alpha_optimizer
    )
    log_alpha = optax.apply_updates(log_alpha, alpha_updates)

    info = {**actor_info, "alpha": jnp.exp(log_alpha), "alpha_loss": alpha_loss}
    return actor_state, log_alpha, alpha_optimizer, info, key


@jax.jit
def soft_update(target_params, online_params, tau):
    """
    Polyak averaging: slowly blend online params into target.

    target = (1 - τ) × target + τ × online

    With τ = 0.005, it takes ~1000 updates for the target to
    approximately match the online network. This provides the
    stable training targets that prevent Q-value divergence.
    """
    return jax.tree.map(
        lambda t, o: (1 - tau) * t + tau * o,
        target_params, online_params
    )


# =========================================================================
# Step 6: MJX Environment Interface
# =========================================================================

def make_env(env_name):
    """Load environment and get observation/action dimensions."""
    env = registry.load(env_name)
    key = jax.random.PRNGKey(0)
    state = env.reset(key)
    obs_dim = state.obs["state"].shape[-1]  # 57 for LeapCubeReorient
    action_dim = env.action_size             # 16 for LeapCubeReorient
    return env, obs_dim, action_dim


# =========================================================================
# Step 7: Training Loop
# =========================================================================

def train(args):
    print("=" * 60)
    print(f"  SAC Training — {args.env_name}")
    print(f"  Seed: {args.seed}")
    print(f"  Num envs: {args.num_envs}")
    print(f"  Total timesteps: {args.total_timesteps:,}")
    print(f"  Buffer size: {args.buffer_size:,}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Hidden dims: {args.hidden_dims}")
    print(f"  LR: {args.learning_rate}, γ: {args.gamma}, τ: {args.tau}")
    print("=" * 60)

    # --- Setup ---
    key = jax.random.PRNGKey(args.seed)
    log_path = os.path.join(args.log_dir, args.run_name)
    os.makedirs(log_path, exist_ok=True)

    # Save config
    with open(os.path.join(log_path, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # TensorBoard
    writer = None
    if args.use_tb:
        try:
            import tensorboardX
            writer = tensorboardX.SummaryWriter(log_path)
        except ImportError:
            print("WARNING: tensorboardX not installed, skipping TB logging")

    # --- Load environment ---
    print("\n[1/5] Loading environment...")
    env, obs_dim, action_dim = make_env(args.env_name)
    print(f"  Obs dim: {obs_dim}, Action dim: {action_dim}")

    # --- Initialize networks ---
    print("[2/5] Initializing networks...")
    key, actor_key, critic_key = jax.random.split(key, 3)

    # Actor
    actor = Actor(action_dim=action_dim, hidden_dims=tuple(args.hidden_dims))
    dummy_obs = jnp.zeros((1, obs_dim))
    actor_params = actor.init(actor_key, dummy_obs)
    actor_state = TrainState.create(
        apply_fn=actor,
        params=actor_params,
        tx=optax.adam(args.learning_rate),
    )

    # Twin Critic
    critic = TwinCritic(hidden_dims=tuple(args.hidden_dims))
    dummy_action = jnp.zeros((1, action_dim))
    critic_params = critic.init(critic_key, dummy_obs, dummy_action)
    critic_state = TrainState.create(
        apply_fn=critic,
        params=critic_params,
        tx=optax.adam(args.learning_rate),
    )
    target_critic_params = critic_params  # Initialize target = online

    # Entropy temperature (α)
    # We optimize log(α) instead of α for numerical stability
    # (α must be positive; log(α) can be any real number)
    target_entropy = -action_dim * args.target_entropy_scale  # -16.0
    log_alpha = jnp.float32(0.0)  # α = exp(0) = 1.0 initially
    alpha_opt_state = optax.adam(args.learning_rate).init(log_alpha)

    num_actor_params = sum(p.size for p in jax.tree.leaves(actor_params))
    num_critic_params = sum(p.size for p in jax.tree.leaves(critic_params))
    print(f"  Actor params: {num_actor_params:,}")
    print(f"  Critic params: {num_critic_params:,}")
    print(f"  Target entropy: {target_entropy}")

    # --- Replay buffer ---
    print("[3/5] Creating replay buffer...")
    buffer = ReplayBuffer(obs_dim, action_dim, args.buffer_size)
    print(f"  Capacity: {args.buffer_size:,} transitions")
    print(f"  Estimated GPU memory: ~{args.buffer_size * (obs_dim*2 + action_dim + 2) * 4 / 1e6:.0f} MB")

    # --- Initialize parallel environments ---
    print("[4/5] Compiling environment functions...")
    key, env_key = jax.random.split(key)

    reset_fn = jax.jit(jax.vmap(env.reset))
    step_fn = jax.jit(jax.vmap(env.step))

    env_keys = jax.random.split(env_key, args.num_envs)
    env_state = reset_fn(env_keys)
    obs = env_state.obs["state"]  # (num_envs, 57)
    print(f"  {args.num_envs} environments ready. Obs shape: {obs.shape}")

    # --- Training loop ---
    print("[5/5] Starting training...")
    print(f"  Random exploration for first {args.learning_starts:,} steps")
    print(f"  JIT compilation will happen on first update (~30s)")
    print()

    global_step = 0
    total_updates = 0
    start_time = time.time()
    metrics_history = []

    while global_step < args.total_timesteps:
        key, action_key = jax.random.split(key)

        # --- Collect one step from all parallel environments ---
        if global_step < args.learning_starts:
            # Random exploration to fill buffer with diverse data
            actions = jax.random.uniform(
                action_key, (args.num_envs, action_dim), minval=-1.0, maxval=1.0
            )
        else:
            # Sample from current policy
            action_keys = jax.random.split(action_key, args.num_envs)
            actions, _ = jax.vmap(
                lambda o, k: sample_action(actor_state.params, actor, o, k)
            )(obs, action_keys)

        # Step all environments
        next_env_state = step_fn(env_state, actions)
        next_obs = next_env_state.obs["state"]
        rewards = next_env_state.reward
        dones = next_env_state.done

        # Store transitions in replay buffer
        buffer.add_batch(obs, actions, rewards, next_obs, dones)

        # Advance
        obs = next_obs
        env_state = next_env_state
        global_step += args.num_envs

        # --- Update networks ---
        if global_step >= args.learning_starts:
            key, sample_key = jax.random.split(key)
            batch = buffer.sample(sample_key, args.batch_size)

            # Update critic
            critic_state, critic_info, key = update_critic(
                critic_state, target_critic_params, actor_state,
                log_alpha, batch, args.gamma, key
            )
            total_updates += 1

            # Delayed actor + alpha update
            if total_updates % args.policy_frequency == 0:
                actor_state, log_alpha, alpha_opt_state, actor_info, key = (
                    update_actor_and_alpha(
                        actor_state, critic_state, log_alpha, alpha_opt_state,
                        batch[0], target_entropy, key  # batch[0] = obs
                    )
                )

                # Soft-update target networks
                target_critic_params = soft_update(
                    target_critic_params, critic_state.params, args.tau
                )

        # --- Logging ---
        if global_step % args.log_every == 0 and global_step > args.learning_starts:
            elapsed = time.time() - start_time
            sps = global_step / elapsed
            alpha_val = float(jnp.exp(log_alpha))

            metrics = {
                "step": global_step,
                "critic_loss": float(critic_info["critic_loss"]),
                "q1_mean": float(critic_info["q1_mean"]),
                "q2_mean": float(critic_info["q2_mean"]),
                "alpha": alpha_val,
                "buffer_size": buffer.size,
                "sps": sps,
            }

            if total_updates % args.policy_frequency == 0:
                metrics["actor_loss"] = float(actor_info["actor_loss"])
                metrics["mean_log_prob"] = float(actor_info["mean_log_prob"])

            metrics_history.append(metrics)

            # TensorBoard logging
            if writer is not None:
                for k, v in metrics.items():
                    writer.add_scalar(k, v, global_step)
                writer.flush()

            print(
                f"Step {global_step:>10,} | "
                f"Critic {float(critic_info['critic_loss']):>8.3f} | "
                f"Q1 {float(critic_info['q1_mean']):>7.1f} | "
                f"α {alpha_val:>6.4f} | "
                f"SPS {sps:>7,.0f} | "
                f"Buf {buffer.size:>8,}"
            )

        # --- Periodic evaluation ---
        if global_step % args.eval_every == 0 and global_step > 0:
            eval_return = evaluate(env, actor_state, actor, key, args.eval_episodes)
            mean_ret = float(eval_return.mean())
            std_ret = float(eval_return.std())

            if writer is not None:
                writer.add_scalar("eval/episode_reward", mean_ret, global_step)
                writer.flush()

            print(f"  >>> EVAL step {global_step:,}: "
                  f"reward = {mean_ret:.2f} ± {std_ret:.2f}")

        # --- Save checkpoint ---
        if global_step % args.save_every == 0 and global_step > 0:
            metrics_path = os.path.join(log_path, f"metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics_history, f)

    # --- Training complete ---
    elapsed = time.time() - start_time
    print(f"\nTraining complete!")
    print(f"  Total time: {elapsed/3600:.2f} hours")
    print(f"  Total steps: {global_step:,}")
    print(f"  Total updates: {total_updates:,}")

    # Final save
    metrics_path = os.path.join(log_path, "metrics_final.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_history, f)
    print(f"  Metrics saved to: {metrics_path}")

    if writer is not None:
        writer.close()


def evaluate(env, actor_state, actor, key, num_episodes):
    """Run deterministic evaluation episodes (mean action, no exploration noise)."""
    returns = []
    for ep in range(num_episodes):
        key, reset_key = jax.random.split(key)
        state = env.reset(reset_key)
        episode_return = 0.0

        for _ in range(1000):
            # Deterministic action: use mean of policy (no sampling noise)
            mean, _ = actor.apply(actor_state.params, state.obs["state"][None])
            action = jnp.tanh(mean[0])  # Squash to [-1, 1]
            state = env.step(state, action)
            episode_return += float(state.reward)
            if state.done:
                break

        returns.append(episode_return)
    return jnp.array(returns)


# =========================================================================
# Entry Point
# =========================================================================

if __name__ == "__main__":
    args = parse_args()
    train(args)