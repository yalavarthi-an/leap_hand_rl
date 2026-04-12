"""
train_sac.py — SAC Training for MuJoCo Playground using Brax's JIT-compiled SAC.

Usage:
  cd ~/leap_hand_rl/mujoco_playground
  uv --no-config run python ../scripts/train_sac.py --seed 0 --num_timesteps 5000000

Author: Anish Yalavarthi (CS 5180 Project)
"""

import datetime
import functools
import json
import os
import time
import warnings

from absl import app
from absl import flags
from absl import logging

from brax.training.agents.sac import networks as sac_networks
from brax.training.agents.sac import train as sac

from etils import epath
import jax
import jax.numpy as jp
import mediapy as media
from ml_collections import config_dict
import mujoco
import mujoco_playground
from mujoco_playground import registry
from mujoco_playground import wrapper

try:
    import tensorboardX
except ImportError:
    tensorboardX = None

try:
    import wandb
except ImportError:
    wandb = None

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

logging.set_verbosity(logging.WARNING)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
warnings.filterwarnings("ignore", category=UserWarning, module="absl")


# =====================================================================
# Flags
# =====================================================================

_ENV_NAME = flags.DEFINE_string("env_name", "LeapCubeReorient", "Environment name")
_IMPL = flags.DEFINE_enum("impl", "jax", ["jax", "warp"], "MJX implementation")
_SUFFIX = flags.DEFINE_string("suffix", None, "Experiment name suffix")
_SEED = flags.DEFINE_integer("seed", 0, "Random seed")
_LOGDIR = flags.DEFINE_string("logdir", None, "Log directory")

_NUM_TIMESTEPS = flags.DEFINE_integer("num_timesteps", 50_000_000, "Total timesteps")
_NUM_ENVS = flags.DEFINE_integer("num_envs", 128, "Parallel environments")
_EPISODE_LENGTH = flags.DEFINE_integer("episode_length", 1000, "Episode length")
_LEARNING_RATE = flags.DEFINE_float("learning_rate", 3e-4, "Learning rate")
_DISCOUNTING = flags.DEFINE_float("discounting", 0.99, "Discount factor")
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 256, "Minibatch size")
_REWARD_SCALING = flags.DEFINE_float("reward_scaling", 1.0, "Reward scaling")
_GRAD_UPDATES_PER_STEP = flags.DEFINE_float("grad_updates_per_step", 1.0, "UTD ratio")
_MAX_REPLAY_SIZE = flags.DEFINE_integer("max_replay_size", 1048576, "Buffer capacity")
_MIN_REPLAY_SIZE = flags.DEFINE_integer("min_replay_size", 8192, "Warmup transitions")
_NORMALIZE_OBSERVATIONS = flags.DEFINE_boolean("normalize_observations", True, "Normalize obs")
_NUM_EVALS = flags.DEFINE_integer("num_evals", 20, "Eval points during training")
_NUM_VIDEOS = flags.DEFINE_integer("num_videos", 1, "Videos after training")
_ACTION_REPEAT = flags.DEFINE_integer("action_repeat", 1, "Action repeat")
_TAU = flags.DEFINE_float("tau", 0.005, "Target network update rate")
_HIDDEN_LAYER_SIZES = flags.DEFINE_list("hidden_layer_sizes", [256, 256, 256], "Network sizes")
_OBS_KEY = flags.DEFINE_string("obs_key", "state", "Obs key: 'state' or 'privileged_state'")
_USE_WANDB = flags.DEFINE_boolean("use_wandb", False, "W&B logging")
_USE_TB = flags.DEFINE_boolean("use_tb", True, "TensorBoard logging")
_DOMAIN_RANDOMIZATION = flags.DEFINE_boolean("domain_randomization", False, "Domain rand")


# =====================================================================
# FlatObsMjxEnv — Wraps the BASE environment to flatten dict observations
#
# WHY AT THE BASE LEVEL:
#   Brax's training wrappers (AutoReset, ActionRepeat, Episode) use
#   jax.lax.scan internally. Scan requires the state pytree structure
#   to be identical between input and output. If reset() returns flat obs
#   but the inner env's step() returns dict obs, scan crashes.
#
#   By wrapping the base MjxEnv BEFORE Brax's wrappers are applied,
#   every level of the wrapper stack sees flat observations consistently.
#
# WRAPPING ORDER:
#   MjxEnv (LeapCubeReorient)        → returns dict obs
#     ↓
#   FlatObsMjxEnv(obs_key='state')   → returns flat obs (57,)
#     ↓
#   wrap_for_brax_training()          → adds AutoReset, Episode tracking
#     ↓
#   Brax SAC train()                  → sees flat obs everywhere ✓
# =====================================================================

class FlatObsMjxEnv:
    """Wraps a Playground MjxEnv to return flat observations from one key."""

    def __init__(self, env, obs_key="state"):
        self._env = env
        self._obs_key = obs_key

    @property
    def observation_size(self):
        """Return int observation size for the selected key."""
        orig = self._env.observation_size
        if isinstance(orig, dict):
            size = orig[self._obs_key]
            if isinstance(size, (tuple, list)):
                return int(size[0])
            return int(size)
        if isinstance(orig, (tuple, list)):
            return int(orig[0])
        return int(orig)

    @property
    def action_size(self):
        size = self._env.action_size
        if isinstance(size, (tuple, list)):
            return int(size[0])
        return int(size)

    def reset(self, rng):
        """Reset and extract selected obs key."""
        state = self._env.reset(rng)
        if isinstance(state.obs, dict):
            return state.replace(obs=state.obs[self._obs_key])
        return state

    def step(self, state, action):
        """Step and extract selected obs key.

        Note: The MjxEnv doesn't use state.obs as input — it recomputes
        observations from the physics state. So passing in a state with
        flat obs (from our reset/previous step) works correctly.
        """
        next_state = self._env.step(state, action)
        if isinstance(next_state.obs, dict):
            return next_state.replace(obs=next_state.obs[self._obs_key])
        return next_state

    def __getattr__(self, name):
        """Forward all other attribute access to the wrapped env.

        This ensures Playground's wrapper can access env-specific attributes
        like dt, sys, mj_model, reward, observation_size, etc.
        """
        return getattr(self._env, name)


# =====================================================================
# SAC Config
# =====================================================================

def get_sac_config(env_name, impl):
    cfg = config_dict.ConfigDict()
    cfg.num_timesteps = 50_000_000
    cfg.num_envs = 128
    cfg.episode_length = 1000
    cfg.action_repeat = 1
    cfg.learning_rate = 3e-4
    cfg.discounting = 0.99
    cfg.batch_size = 256
    cfg.reward_scaling = 1.0
    cfg.grad_updates_per_step = 1.0
    cfg.tau = 0.005
    cfg.max_replay_size = 1048576
    cfg.min_replay_size = 8192
    cfg.normalize_observations = True
    cfg.num_evals = 20
    cfg.network_factory = config_dict.ConfigDict({
        "hidden_layer_sizes": [256, 256, 256],
        "obs_key": "state",
    })
    return cfg


# =====================================================================
# Main
# =====================================================================

def main(argv):
    del argv

    env_cfg = registry.get_default_config(_ENV_NAME.value)
    env_cfg["impl"] = _IMPL.value

    sac_params = get_sac_config(_ENV_NAME.value, _IMPL.value)

    # Apply overrides
    if _NUM_TIMESTEPS.present: sac_params.num_timesteps = _NUM_TIMESTEPS.value
    if _NUM_ENVS.present: sac_params.num_envs = _NUM_ENVS.value
    if _EPISODE_LENGTH.present: sac_params.episode_length = _EPISODE_LENGTH.value
    if _LEARNING_RATE.present: sac_params.learning_rate = _LEARNING_RATE.value
    if _DISCOUNTING.present: sac_params.discounting = _DISCOUNTING.value
    if _BATCH_SIZE.present: sac_params.batch_size = _BATCH_SIZE.value
    if _REWARD_SCALING.present: sac_params.reward_scaling = _REWARD_SCALING.value
    if _GRAD_UPDATES_PER_STEP.present: sac_params.grad_updates_per_step = _GRAD_UPDATES_PER_STEP.value
    if _MAX_REPLAY_SIZE.present: sac_params.max_replay_size = _MAX_REPLAY_SIZE.value
    if _MIN_REPLAY_SIZE.present: sac_params.min_replay_size = _MIN_REPLAY_SIZE.value
    if _NORMALIZE_OBSERVATIONS.present: sac_params.normalize_observations = _NORMALIZE_OBSERVATIONS.value
    if _NUM_EVALS.present: sac_params.num_evals = _NUM_EVALS.value
    if _ACTION_REPEAT.present: sac_params.action_repeat = _ACTION_REPEAT.value
    if _TAU.present: sac_params.tau = _TAU.value
    if _HIDDEN_LAYER_SIZES.present:
        sac_params.network_factory.hidden_layer_sizes = list(map(int, _HIDDEN_LAYER_SIZES.value))
    if _OBS_KEY.present: sac_params.network_factory.obs_key = _OBS_KEY.value

    # Load environment
    env = registry.load(_ENV_NAME.value, config=env_cfg)

    print(f"Environment Config:\n{env_cfg}")
    print(f"\nSAC Training Parameters:\n{sac_params}")

    # Experiment name
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    exp_name = f"SAC-{_ENV_NAME.value}-{timestamp}"
    if _SUFFIX.value: exp_name += f"-{_SUFFIX.value}"
    print(f"Experiment name: {exp_name}")

    # Logging setup
    logdir = epath.Path(_LOGDIR.value or "logs").resolve() / exp_name
    logdir.mkdir(parents=True, exist_ok=True)
    print(f"Logs: {logdir}")

    ckpt_path = logdir / "checkpoints"
    ckpt_path.mkdir(parents=True, exist_ok=True)

    with open(ckpt_path / "config.json", "w", encoding="utf-8") as fp:
        json.dump(env_cfg.to_dict(), fp, indent=4)

    writer = None
    if _USE_TB.value and tensorboardX is not None:
        writer = tensorboardX.SummaryWriter(logdir)

    if _USE_WANDB.value:
        if wandb is None: raise ImportError("wandb required")
        wandb.init(project="mjxrl-sac", name=exp_name)
        wandb.config.update(env_cfg.to_dict())

    # ============================================================
    # Build training function
    # ============================================================

    training_params = dict(sac_params)
    del training_params["network_factory"]

    # Ensure integer params stay as ints (ConfigDict can convert to float)
    for key in ["batch_size", "num_envs", "num_timesteps", "episode_length",
                "max_replay_size", "min_replay_size", "num_evals", "action_repeat",
                "grad_updates_per_step", "tau"]:
        if key in training_params:
            training_params[key] = int(training_params[key])

    obs_key = sac_params.network_factory.get("obs_key", "state")
    hidden_sizes = list(map(int, sac_params.network_factory.get(
        "hidden_layer_sizes", [256, 256, 256]
    )))

    network_factory = functools.partial(
        sac_networks.make_sac_networks,
        hidden_layer_sizes=hidden_sizes,
    )

    if _DOMAIN_RANDOMIZATION.value:
        training_params["randomization_fn"] = registry.get_domain_randomizer(
            _ENV_NAME.value
        )

    # ============================================================
    # wrap_env_fn: Flatten obs at base level THEN apply Brax wrappers
    # This is the key fix — flattening happens BEFORE jax.lax.scan
    # ============================================================
    def wrap_env_for_sac(env, **kwargs):
        flat_env = FlatObsMjxEnv(env, obs_key=obs_key)
        return wrapper.wrap_for_brax_training(flat_env, **kwargs)

    train_fn = functools.partial(
        sac.train,
        **training_params,
        network_factory=network_factory,
        seed=_SEED.value,
        wrap_env_fn=wrap_env_for_sac,
    )

    # ============================================================
    # Progress callback
    # ============================================================
    times = [time.monotonic()]

    def progress(num_steps, metrics):
        times.append(time.monotonic())
        if _USE_TB.value and writer is not None:
            for key, value in metrics.items():
                writer.add_scalar(key, value, num_steps)
            writer.flush()
        if _USE_WANDB.value:
            wandb.log(metrics, step=num_steps)
        reward = metrics.get("eval/episode_reward", 0)
        print(f"{num_steps}: reward={reward:.3f}")

    # ============================================================
    # TRAIN
    # ============================================================
    print("\nStarting SAC training...")
    print("JIT compilation will take 1-3 minutes...")

    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=progress,
    )

    print("\nDone training.")
    if len(times) > 1:
        print(f"Time to JIT compile: {times[1] - times[0]:.1f}s")
        print(f"Time to train: {times[-1] - times[1]:.1f}s")

    # ============================================================
    # Post-training video
    # ============================================================
    print("\nStarting inference...")

    inference_fn = make_inference_fn(params, deterministic=True)
    jit_inference_fn = jax.jit(inference_fn)

    # Use FlatObsMjxEnv for inference too
    infer_env_raw = registry.load(_ENV_NAME.value, config=env_cfg)
    infer_env_flat = FlatObsMjxEnv(infer_env_raw, obs_key=obs_key)
    wrapped_infer_env = wrapper.wrap_for_brax_training(
        infer_env_flat,
        episode_length=sac_params.episode_length,
        action_repeat=sac_params.get("action_repeat", 1),
    )

    rng = jax.random.split(jax.random.PRNGKey(_SEED.value), _NUM_VIDEOS.value)
    reset_states = jax.jit(wrapped_infer_env.reset)(rng)

    empty_data = reset_states.data.__class__(
        **{k: None for k in reset_states.data.__annotations__}
    )
    empty_traj = reset_states.__class__(
        **{k: None for k in reset_states.__annotations__}
    )
    empty_traj = empty_traj.replace(data=empty_data)

    def step(carry, _):
        state, rng = carry
        rng, act_key = jax.random.split(rng)
        act_keys = jax.random.split(act_key, _NUM_VIDEOS.value)
        act = jax.vmap(jit_inference_fn)(state.obs, act_keys)[0]
        state = wrapped_infer_env.step(state, act)
        traj_data = empty_traj.tree_replace({
            "data.qpos": state.data.qpos,
            "data.qvel": state.data.qvel,
            "data.time": state.data.time,
            "data.ctrl": state.data.ctrl,
            "data.mocap_pos": state.data.mocap_pos,
            "data.mocap_quat": state.data.mocap_quat,
            "data.xfrc_applied": state.data.xfrc_applied,
        })
        return (state, rng), traj_data

    @jax.jit
    def do_rollout(state, rng):
        _, traj = jax.lax.scan(step, (state, rng), None, length=sac_params.episode_length)
        return traj

    traj_stacked = do_rollout(reset_states, jax.random.PRNGKey(_SEED.value + 1))
    traj_stacked = jax.tree.map(lambda x: jp.moveaxis(x, 0, 1), traj_stacked)
    trajectories = [None] * _NUM_VIDEOS.value
    for i in range(_NUM_VIDEOS.value):
        t = jax.tree.map(lambda x, i=i: x[i], traj_stacked)
        trajectories[i] = [jax.tree.map(lambda x, j=j: x[j], t) for j in range(sac_params.episode_length)]

    render_every = 2
    fps = 1.0 / infer_env_raw.dt / render_every
    print(f"FPS: {fps}")
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False

    for i in range(_NUM_VIDEOS.value):
        frames = infer_env_raw.render(trajectories[i][::render_every], scene_option=scene_option)
        media.write_video(logdir / f"rollout{i}.mp4", frames, fps=fps)
        print(f"Video saved: {logdir / f'rollout{i}.mp4'}")

    if writer is not None:
        writer.close()


def run():
    app.run(main)

if __name__ == "__main__":
    run()