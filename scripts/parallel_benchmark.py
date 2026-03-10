"""
parallel_benchmark.py — Run 1024 environments in parallel on GPU.

This is THE core concept that makes GPU-accelerated RL possible.
Instead of stepping one environment at a time (CPU-style), we step
1024 environments simultaneously on the GPU.

Key concepts:

  jax.vmap(fn):
    "Vectorized map" — transforms a function that operates on a SINGLE
    input into one that operates on a BATCH of inputs. You write code
    for one environment, and vmap automatically creates the batched version.

    Example:
      env.reset(one_key)          → resets 1 environment
      jax.vmap(env.reset)(keys)   → resets 1024 environments at once

  jax.jit(fn):
    "Just-In-Time compilation" — traces the Python function, compiles it
    into an optimized GPU kernel via XLA, and caches the result.
    - First call: SLOW (compilation, 10-30 seconds for complex functions)
    - All subsequent calls: FAST (runs the cached GPU kernel)

  jax.block_until_ready(x):
    JAX operations are asynchronous — jax.jit returns immediately and
    the GPU runs in the background. block_until_ready forces Python to
    wait until the GPU finishes, so our timing measurements are accurate.

  jax.random.split(key, n):
    Splits one PRNG key into n independent keys. Each parallel environment
    needs its own key to generate different random initial states.

  XLA_FLAGS:
    Environment variable that controls XLA's compiler behavior.
    --xla_gpu_triton_gemm_any=true enables Triton-based matrix multiplication
    kernels which can significantly speed up computation on NVIDIA GPUs.

Usage:
  cd ~/leap_hand_rl/mujoco_playground
  uv --no-config run python ../scripts/parallel_benchmark.py
"""

import os

# Set XLA optimization flag BEFORE importing JAX
# This must be done before any JAX import to take effect
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=true"

import jax
import jax.numpy as jnp
import time
from mujoco_playground import registry

NUM_ENVS = 1024
NUM_WARMUP_STEPS = 1   # For JIT compilation
NUM_BENCHMARK_STEPS = 200

print("=" * 60)
print(f"  Parallel Environment Benchmark — {NUM_ENVS} Environments")
print("=" * 60)

# Load environment
env = registry.load("LeapCubeReorient")

# Create 1024 independent random keys (one per environment)
key = jax.random.PRNGKey(0)
keys = jax.random.split(key, NUM_ENVS)

# === The magic of vmap + jit ===
#
# env.reset handles ONE environment.
# jax.vmap(env.reset) handles a BATCH of environments.
# jax.jit(...) compiles it into a single optimized GPU kernel.
#
# The combined function resets all 1024 environments in ONE GPU call.
reset_fn = jax.jit(jax.vmap(env.reset))
step_fn = jax.jit(jax.vmap(env.step))

# --- Reset all environments ---
print(f"\n[1/3] Resetting {NUM_ENVS} environments (includes JIT compilation)...")
t0 = time.time()
states = reset_fn(keys)
jax.block_until_ready(states.obs["state"])  # Wait for GPU to finish
t_reset = time.time() - t0
print(f"  Reset time     : {t_reset:.2f}s (includes JIT compilation)")
print(f"  Batch obs shape: {states.obs['state'].shape}")  # Should be (1024, 57)

# --- First step (triggers JIT compilation of step function) ---
actions = jax.random.uniform(key, (NUM_ENVS, 16), minval=-1.0, maxval=1.0)

print(f"\n[2/3] First step (includes JIT compilation of step function)...")
t0 = time.time()
next_states = step_fn(states, actions)
jax.block_until_ready(next_states.obs["state"])
t_first = time.time() - t0
print(f"  First step     : {t_first:.2f}s (mostly compilation)")

# --- Benchmark: compiled speed ---
print(f"\n[3/3] Benchmarking {NUM_BENCHMARK_STEPS} steps (full compiled speed)...")
t0 = time.time()
for i in range(NUM_BENCHMARK_STEPS):
    next_states = step_fn(next_states, actions)
jax.block_until_ready(next_states.obs["state"])
t_fast = time.time() - t0

throughput = NUM_ENVS * NUM_BENCHMARK_STEPS / t_fast

print(f"  {NUM_BENCHMARK_STEPS} steps      : {t_fast:.2f}s")
print(f"  Per step       : {t_fast / NUM_BENCHMARK_STEPS * 1000:.1f}ms")
print(f"  Throughput     : {throughput:,.0f} env steps/second")

# --- Summary statistics ---
print(f"\n  Rewards — mean: {float(next_states.reward.mean()):.4f}, "
      f"min: {float(next_states.reward.min()):.4f}, "
      f"max: {float(next_states.reward.max()):.4f}")
print(f"  Episodes done  : {int(next_states.done.sum())} / {NUM_ENVS}")

# --- Training time estimates ---
print(f"\n  Estimated training times at {throughput:,.0f} steps/sec:")
for total_steps, label in [
    (10_000_000, "10M steps (quick test)"),
    (50_000_000, "50M steps (medium run)"),
    (100_000_000, "100M steps (full run)"),
]:
    hours = total_steps / throughput / 3600
    print(f"    {label}: ~{hours:.1f} hours")