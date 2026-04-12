"""
train_sac_v3.py — Soft Actor-Critic for LeapCubeReorient (MJX)

A complete JAX/Flax SAC implementation designed specifically for
contact-rich dexterous manipulation with MuJoCo Playground.

Key improvements over v1/v2:
  1. Observation normalization (running mean/std via Welford's algorithm)
  2. Asymmetric actor-critic (actor: state 57d, critic: privileged_state 128d)
  3. Layer normalization on Q-networks (prevents Q-value collapse — RLPD)
  4. Reward scaling (×0.1) for numerical stability
  5. UTD ratio (4 gradient updates per env step)
  6. Alpha clamping (min α = 0.01 prevents entropy collapse)
  7. JIT-compiled evaluation (jax.lax.scan, not slow Python loops)
  8. Normalized observations stored in buffer for consistent Q-learning

Usage:
  cd ~/leap_hand_rl/mujoco_playground
  uv --no-config run python ../scripts/train_sac_v3.py --seed 0

Author: Anish Yalavarthi (CS 5180 Project)
"""

import os
import time
import json
import argparse
from datetime import datetime
from functools import partial

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training.train_state import TrainState
from mujoco_playground import registry


# =========================================================================
# 1. CONFIGURATION
# =========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="SAC v3 for LeapCubeReorient")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--log_dir", type=str, default="./logs")
    p.add_argument("--env_name", type=str, default="LeapCubeReorient")
    p.add_argument("--num_envs", type=int, default=256)
    p.add_argument("--total_timesteps", type=int, default=50_000_000)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--buffer_size", type=int, default=1_000_000)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--warmup_steps", type=int, default=10000)
    p.add_argument("--policy_frequency", type=int, default=2)
    p.add_argument("--reward_scale", type=float, default=0.1)
    p.add_argument("--utd_ratio", type=int, default=4)
    p.add_argument("--alpha_min", type=float, default=0.01)
    p.add_argument("--alpha_lr", type=float, default=3e-4)
    p.add_argument("--actor_hidden", type=int, nargs="+", default=[256, 256])
    p.add_argument("--critic_hidden", type=int, nargs="+", default=[256, 256, 256])
    p.add_argument("--log_every", type=int, default=10000)
    p.add_argument("--eval_every", type=int, default=5_000_000)
    p.add_argument("--eval_episodes", type=int, default=8)
    p.add_argument("--save_every", type=int, default=5_000_000)
    p.add_argument("--use_tb", action="store_true")
    args = p.parse_args()
    if args.run_name is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"sacv3_{args.env_name}_s{args.seed}_{ts}"
    return args


# =========================================================================
# 2. OBSERVATION NORMALIZATION (Welford's Online Algorithm)
#
# The 57 obs dims have wildly different scales:
#   joint angles ~0-1.5, velocities ~±10, quaternions ~±1
# Without normalization, the network can't learn uniformly.
# After normalization: all dims have mean≈0, std≈1.
# =========================================================================

class RunningMeanStd:
    def __init__(self, shape):
        self.mean = jnp.zeros(shape)
        self.var = jnp.ones(shape)
        self.count = 1e-4

    def update(self, batch):
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta**2 * self.count * batch_count / total) / total
        self.count = total

    def normalize(self, obs):
        return (obs - self.mean) / jnp.sqrt(self.var + 1e-8)


# =========================================================================
# 3. ACTOR NETWORK — sees 'state' (57d) only
# =========================================================================

class Actor(nn.Module):
    action_dim: int
    hidden_dims: tuple = (256, 256)
    LOG_STD_MIN: float = -20.0
    LOG_STD_MAX: float = 2.0

    @nn.compact
    def __call__(self, obs):
        x = obs
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std


def sample_action(params, actor, obs, key):
    mean, log_std = actor.apply(params, obs)
    std = jnp.exp(log_std)
    noise = jax.random.normal(key, mean.shape)
    u = mean + std * noise
    action = jnp.tanh(u)
    log_prob = -0.5 * (((u - mean) / (std + 1e-8))**2 + 2*log_std + jnp.log(2*jnp.pi))
    log_prob = log_prob.sum(axis=-1)
    log_prob -= jnp.sum(jnp.log(1 - action**2 + 1e-6), axis=-1)
    return action, log_prob


# =========================================================================
# 4. TWIN CRITIC with LAYER NORMALIZATION — sees 'privileged_state' (128d)
#
# LayerNorm after each hidden layer prevents Q-value extrapolation errors.
# This is the single most impactful architectural choice for contact-rich
# tasks according to RLPD (Ball et al., ICML 2023).
# =========================================================================

class TwinCritic(nn.Module):
    hidden_dims: tuple = (256, 256, 256)

    @nn.compact
    def __call__(self, obs, action):
        x = jnp.concatenate([obs, action], axis=-1)
        q1 = x
        for dim in self.hidden_dims:
            q1 = nn.Dense(dim)(q1)
            q1 = nn.LayerNorm()(q1)
            q1 = nn.relu(q1)
        q1 = nn.Dense(1)(q1).squeeze(-1)
        q2 = x
        for dim in self.hidden_dims:
            q2 = nn.Dense(dim)(q2)
            q2 = nn.LayerNorm()(q2)
            q2 = nn.relu(q2)
        q2 = nn.Dense(1)(q2).squeeze(-1)
        return q1, q2


# =========================================================================
# 5. REPLAY BUFFER — stores both state and privileged_state
# =========================================================================

class ReplayBuffer:
    def __init__(self, state_dim, priv_dim, action_dim, capacity):
        self.capacity = capacity
        self.state = jnp.zeros((capacity, state_dim))
        self.priv = jnp.zeros((capacity, priv_dim))
        self.actions = jnp.zeros((capacity, action_dim))
        self.rewards = jnp.zeros((capacity,))
        self.next_state = jnp.zeros((capacity, state_dim))
        self.next_priv = jnp.zeros((capacity, priv_dim))
        self.dones = jnp.zeros((capacity,))
        self.size = 0
        self.ptr = 0

    def add_batch(self, state, priv, actions, rewards, next_state, next_priv, dones):
        bs = state.shape[0]
        idx = (jnp.arange(bs) + self.ptr) % self.capacity
        self.state = self.state.at[idx].set(state)
        self.priv = self.priv.at[idx].set(priv)
        self.actions = self.actions.at[idx].set(actions)
        self.rewards = self.rewards.at[idx].set(rewards)
        self.next_state = self.next_state.at[idx].set(next_state)
        self.next_priv = self.next_priv.at[idx].set(next_priv)
        self.dones = self.dones.at[idx].set(dones)
        self.ptr = (self.ptr + bs) % self.capacity
        self.size = min(self.size + bs, self.capacity)

    def sample(self, key, batch_size):
        idx = jax.random.randint(key, (batch_size,), 0, self.size)
        return (self.state[idx], self.priv[idx], self.actions[idx],
                self.rewards[idx], self.next_state[idx], self.next_priv[idx],
                self.dones[idx])


# =========================================================================
# 6. SAC UPDATE FUNCTIONS (all JIT-compiled)
# =========================================================================

@partial(jax.jit, static_argnums=(3, 4))
def update_critic(
    critic_state, target_critic_params, actor_state,
    actor, critic, log_alpha,
    actor_obs, critic_obs, actions, rewards,
    next_actor_obs, next_critic_obs, dones, gamma, key,
):
    key, ak = jax.random.split(key)
    next_actions, next_lp = sample_action(actor_state.params, actor, next_actor_obs, ak)
    alpha = jnp.exp(log_alpha)
    nq1, nq2 = critic.apply(target_critic_params, next_critic_obs, next_actions)
    next_q = jnp.minimum(nq1, nq2) - alpha * next_lp
    target_q = jax.lax.stop_gradient(rewards + gamma * (1.0 - dones) * next_q)

    def loss_fn(params):
        q1, q2 = critic.apply(params, critic_obs, actions)
        loss = 0.5 * (jnp.mean((q1 - target_q)**2) + jnp.mean((q2 - target_q)**2))
        return loss, {"critic_loss": loss, "q1_mean": q1.mean(),
                       "q2_mean": q2.mean(), "target_q": target_q.mean()}

    (_, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(critic_state.params)
    critic_state = critic_state.apply_gradients(grads=grads)
    return critic_state, info, key


@partial(jax.jit, static_argnums=(1, 3))
def update_actor(
    actor_state, actor, critic_state, critic,
    log_alpha, actor_obs, critic_obs, key,
):
    key, ak = jax.random.split(key)
    alpha = jnp.exp(log_alpha)

    def loss_fn(params):
        actions, lp = sample_action(params, actor, actor_obs, ak)
        q1, q2 = critic.apply(critic_state.params, critic_obs, actions)
        loss = jnp.mean(alpha * lp - jnp.minimum(q1, q2))
        return loss, {"actor_loss": loss, "mean_log_prob": lp.mean(),
                       "mean_std": jnp.exp(actor.apply(params, actor_obs)[1]).mean()}

    (_, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(actor_state.params)
    actor_state = actor_state.apply_gradients(grads=grads)
    return actor_state, info, key


@jax.jit
def update_alpha(log_alpha, mean_log_prob, target_entropy, lr, min_log):
    grad = -(mean_log_prob + target_entropy)
    log_alpha = log_alpha - lr * grad
    log_alpha = jnp.maximum(log_alpha, min_log)
    return log_alpha


@jax.jit
def soft_update(target, online, tau):
    return jax.tree.map(lambda t, o: (1 - tau) * t + tau * o, target, online)


# =========================================================================
# 7. JIT-COMPILED EVALUATION
# =========================================================================

def make_eval_fn(env, actor, episode_length=1000):
    @jax.jit
    def evaluate(actor_params, norm_mean, norm_var, key, num_envs):
        keys = jax.random.split(key, num_envs)
        state = jax.vmap(env.reset)(keys)

        def step_fn(carry, _):
            state, cum_rew, active, rng = carry
            rng, ak = jax.random.split(rng)
            obs = (state.obs["state"] - norm_mean) / jnp.sqrt(norm_var + 1e-8)
            mean, _ = jax.vmap(lambda o: actor.apply(actor_params, o))(obs)
            actions = jnp.tanh(mean)
            next_state = jax.vmap(env.step)(state, actions)
            cum_rew = cum_rew + active * next_state.reward
            active = active * (1.0 - next_state.done)
            return (next_state, cum_rew, active, rng), None

        carry = (state, jnp.zeros(num_envs), jnp.ones(num_envs), key)
        (_, cum_rew, _, _), _ = jax.lax.scan(step_fn, carry, None, length=episode_length)
        return cum_rew

    return evaluate


# =========================================================================
# 8. TRAINING LOOP
# =========================================================================

def train(args):
    print("=" * 60)
    print(f"  SAC v3 — {args.env_name}")
    print(f"  Obs norm ✓ | Asymmetric critic ✓ | LayerNorm ✓")
    print(f"  Seed: {args.seed}, Envs: {args.num_envs}")
    print(f"  Steps: {args.total_timesteps:,}, Batch: {args.batch_size}")
    print(f"  UTD: {args.utd_ratio}, Reward scale: {args.reward_scale}")
    print(f"  Actor: {args.actor_hidden}, Critic: {args.critic_hidden}+LN")
    print("=" * 60)

    key = jax.random.PRNGKey(args.seed)
    log_path = os.path.join(args.log_dir, args.run_name)
    os.makedirs(log_path, exist_ok=True)
    with open(os.path.join(log_path, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    writer = None
    if args.use_tb:
        try:
            import tensorboardX
            writer = tensorboardX.SummaryWriter(log_path)
        except ImportError:
            print("WARNING: tensorboardX not installed")

    # --- Load environment ---
    print("\n[1/6] Loading environment...")
    env = registry.load(args.env_name)
    key, rk = jax.random.split(key)
    dummy = env.reset(rk)
    state_dim = dummy.obs["state"].shape[-1]
    priv_dim = dummy.obs["privileged_state"].shape[-1]
    action_dim = env.action_size
    print(f"  Actor input: state ({state_dim}d)")
    print(f"  Critic input: privileged_state ({priv_dim}d) + action ({action_dim}d)")

    # --- Networks ---
    print("[2/6] Initializing networks...")
    key, ak, ck = jax.random.split(key, 3)
    actor = Actor(action_dim=action_dim, hidden_dims=tuple(args.actor_hidden))
    actor_state = TrainState.create(
        apply_fn=actor, params=actor.init(ak, jnp.zeros((1, state_dim))),
        tx=optax.adam(args.learning_rate))

    critic = TwinCritic(hidden_dims=tuple(args.critic_hidden))
    critic_params = critic.init(ck, jnp.zeros((1, priv_dim)), jnp.zeros((1, action_dim)))
    critic_state = TrainState.create(
        apply_fn=critic, params=critic_params, tx=optax.adam(args.learning_rate))
    target_critic_params = critic_params

    target_entropy = -float(action_dim)
    log_alpha = jnp.float32(0.0)
    alpha_min_log = jnp.log(jnp.float32(args.alpha_min))

    n_a = sum(p.size for p in jax.tree.leaves(actor_state.params))
    n_c = sum(p.size for p in jax.tree.leaves(critic_params))
    print(f"  Actor: {n_a:,} params | Critic: {n_c:,} params")

    # --- Normalizers ---
    print("[3/6] Observation normalizers...")
    state_norm = RunningMeanStd(state_dim)
    priv_norm = RunningMeanStd(priv_dim)

    # --- Buffer ---
    print("[4/6] Replay buffer...")
    buffer = ReplayBuffer(state_dim, priv_dim, action_dim, args.buffer_size)

    # --- Environments ---
    print("[5/6] Compiling env functions...")
    key, ek = jax.random.split(key)
    reset_fn = jax.jit(jax.vmap(env.reset))
    step_fn = jax.jit(jax.vmap(env.step))
    env_state = reset_fn(jax.random.split(ek, args.num_envs))
    obs_s = env_state.obs["state"]
    obs_p = env_state.obs["privileged_state"]

    eval_fn = make_eval_fn(env, actor)
    print(f"  {args.num_envs} envs ready")

    # --- Train ---
    print("[6/6] Training...\n")
    global_step = 0
    total_updates = 0
    t0 = time.time()
    history = []
    ci = {"critic_loss": 0., "q1_mean": 0., "q2_mean": 0., "target_q": 0.}
    ai = {"actor_loss": 0., "mean_log_prob": 0., "mean_std": 1.}

    while global_step < args.total_timesteps:
        key, action_key = jax.random.split(key)

        if global_step < args.warmup_steps:
            actions = jax.random.uniform(action_key, (args.num_envs, action_dim), minval=-1., maxval=1.)
        else:
            obs_n = state_norm.normalize(obs_s)
            aks = jax.random.split(action_key, args.num_envs)
            actions, _ = jax.vmap(lambda o, k: sample_action(actor_state.params, actor, o, k))(obs_n, aks)

        next_env = step_fn(env_state, actions)
        n_s = next_env.obs["state"]
        n_p = next_env.obs["privileged_state"]
        rewards = next_env.reward
        dones = next_env.done

        state_norm.update(n_s)
        priv_norm.update(n_p)

        buffer.add_batch(
            state_norm.normalize(obs_s), priv_norm.normalize(obs_p),
            actions, rewards * args.reward_scale,
            state_norm.normalize(n_s), priv_norm.normalize(n_p), dones)

        obs_s, obs_p, env_state = n_s, n_p, next_env
        global_step += args.num_envs

        if global_step >= args.warmup_steps and buffer.size >= args.batch_size:
            for _ in range(args.utd_ratio):
                key, sk = jax.random.split(key)
                b = buffer.sample(sk, args.batch_size)

                critic_state, ci, key = update_critic(
                    critic_state, target_critic_params, actor_state,
                    actor, critic, log_alpha,
                    b[0], b[1], b[2], b[3], b[4], b[5], b[6],
                    args.gamma, key)
                total_updates += 1

                if total_updates % args.policy_frequency == 0:
                    actor_state, ai, key = update_actor(
                        actor_state, actor, critic_state, critic,
                        log_alpha, b[0], b[1], key)
                    log_alpha = update_alpha(
                        log_alpha, ai["mean_log_prob"],
                        target_entropy, args.alpha_lr, alpha_min_log)
                    target_critic_params = soft_update(
                        target_critic_params, critic_state.params, args.tau)

        if global_step % args.log_every == 0 and global_step > args.warmup_steps:
            el = time.time() - t0
            sps = global_step / el
            av = float(jnp.exp(log_alpha))
            m = {"step": global_step, "critic_loss": float(ci["critic_loss"]),
                 "q1": float(ci["q1_mean"]), "alpha": av,
                 "std": float(ai["mean_std"]), "R": float(rewards.mean()),
                 "sps": sps, "upd": total_updates}
            history.append(m)
            if writer:
                for k, v in m.items():
                    writer.add_scalar(k, v, global_step)
                writer.flush()
            print(f"Step {global_step:>10,} | Crit {float(ci['critic_loss']):>7.3f} | "
                  f"Q1 {float(ci['q1_mean']):>7.2f} | α {av:>6.4f} | "
                  f"Std {float(ai['mean_std']):>5.3f} | R {float(rewards.mean()):>7.3f} | "
                  f"SPS {sps:>6,.0f}")

        if global_step % args.eval_every == 0 and global_step > 0:
            key, evk = jax.random.split(key)
            er = eval_fn(actor_state.params, state_norm.mean, state_norm.var, evk, args.eval_episodes)
            mr, sr = float(er.mean()), float(er.std())
            if writer:
                writer.add_scalar("eval/episode_reward", mr, global_step)
            print(f"  >>> EVAL {global_step:,}: {mr:.1f} ± {sr:.1f}")

        if global_step % args.save_every == 0 and global_step > 0:
            with open(os.path.join(log_path, "metrics.json"), "w") as f:
                json.dump(history, f)

    el = time.time() - t0
    print(f"\nDone! {el/3600:.2f}h, {global_step:,} steps, {total_updates:,} updates")
    print(f"Final: α={float(jnp.exp(log_alpha)):.4f}, Q1={float(ci['q1_mean']):.2f}")
    with open(os.path.join(log_path, "metrics_final.json"), "w") as f:
        json.dump(history, f)
    if writer:
        writer.close()


if __name__ == "__main__":
    train(parse_args())