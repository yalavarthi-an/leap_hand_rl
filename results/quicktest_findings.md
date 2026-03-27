# Preliminary Experimental Findings — Quick Tests

## CS 5180 Project: PPO vs SAC on LEAP Hand
## Author: Anish Yalavarthi
## Date: March 26, 2026

---

## 1. Hardware Configuration

- **GPU:** NVIDIA GeForce RTX 5070 Laptop (8,151 MiB VRAM)
- **Architecture:** Blackwell
- **CUDA Version:** 13.1
- **Driver:** 590.48.01
- **CPU:** 16 cores
- **RAM:** 33 GB (30 GB usable)
- **OS:** Ubuntu 24.04

## 2. Software Stack

- **Python:** 3.12.12
- **JAX:** 0.6.2
- **MuJoCo Playground:** installed from source (March 2026)
- **Physics backend:** MJX-JAX (`--impl jax`)
- **Precision:** `JAX_DEFAULT_MATMUL_PRECISION=highest` (full float32)
- **XLA flags:** `--xla_gpu_triton_gemm_any=true` (Triton GEMM optimization)

## 3. Environment Details

- **Environment:** LeapCubeReorient (MuJoCo Playground)
- **Observation space:** `state` (57 dims), `privileged_state` (128 dims)
- **Action space:** 16 continuous (joint target positions)
- **Sim timestep (dt):** 0.01s
- **Control timestep (ctrl_dt):** 0.05s
- **Episode length:** 1000 steps
- **Success threshold:** 0.1 radians

### Reward Structure (from env config)
| Component       | Weight  | Description                              |
|----------------|---------|------------------------------------------|
| orientation    | +5.0    | Cube orientation close to target         |
| position       | +0.5    | Cube position stability                  |
| success_reward | +100.0  | Bonus when within success_threshold      |
| hand_pose      | -0.5    | Penalty for unnatural hand configuration |
| termination    | -100.0  | Penalty when cube is dropped             |
| action_rate    | -0.001  | Penalty for jerky actions                |
| energy         | -0.001  | Penalty for high energy usage            |

## 4. PPO Configuration (Pre-tuned by Playground developers)

| Parameter                    | Value           | Notes                          |
|-----------------------------|-----------------|--------------------------------|
| Network (actor)             | (512, 256, 128) | Tapered MLP architecture       |
| Network (critic)            | (512, 256, 128) | Same as actor                  |
| Actor obs key               | `state` (57d)   | Only real-robot-measurable obs |
| Critic obs key              | `privileged_state` (128d) | Asymmetric — sim-only info |
| Learning rate               | 3e-4            |                                |
| Discount (γ)                | 0.99            |                                |
| Clipping (ε)                | 0.2             | PPO clip range                 |
| Entropy cost                | 0.01            |                                |
| Unroll length               | 40              | Steps per collection window    |
| Num updates per batch       | 4               | Epochs over each batch         |
| Num minibatches             | 32              |                                |
| Reward scaling              | 1.0             | No scaling                     |
| Observation normalization   | True            | Running mean/std               |

## 5. GPU Benchmark: Raw Environment Throughput

Tested with `jax.vmap(env.step)` on 1024 environments, 200 steps:

| Metric                | Without Triton | With Triton Flag |
|-----------------------|---------------|------------------|
| Steps/second          | 4,051         | 7,176            |
| Per-step latency      | 252.8 ms      | 142.7 ms         |
| Speedup               | baseline      | 1.77×            |

**Note:** The `--xla_gpu_triton_gemm_any=true` flag provides ~77% throughput
improvement on the RTX 5070 Blackwell GPU for this environment.

## 6. Parallel Environment Scaling (Quick Test: 2M timesteps each)

| Num Envs | VRAM Used | JIT Time | Train Time | Final Reward | Throughput* |
|----------|-----------|----------|------------|-------------|-------------|
| 1024     | ~2.4 GB   | 51.4s    | 535.6s     | 114.1       | ~11,600 sps |
| 2048     | ~2.5 GB   | 58.2s    | 551.1s     | 116.7       | ~11,300 sps |
| 4096     | ~3.0 GB   | 57.9s    | 545.1s     | 114.9       | ~11,400 sps |

*Throughput = effective env steps / training time (excludes JIT)

### Key Finding: Fixed Contact Buffer Dominates VRAM

The `naconmax: 245760` parameter pre-allocates a fixed-size contact buffer
(~1.5 GB) regardless of num_envs. This is due to MJX-JAX's SIMD constraint
requiring static array shapes at compile time. Consequence:

- Doubling num_envs from 1024→4096 only increased VRAM by ~0.6 GB
- Training wall-clock time was nearly identical across all three settings
- The contact buffer, not the environment count, is the VRAM bottleneck
  for contact-rich dexterous manipulation tasks

**Implication for the paper:** On consumer GPUs, the practical benefit of
increasing parallel environments is bounded by the fixed contact allocation.
This contrasts with simpler environments (CartPole, locomotion) where VRAM
scales linearly with num_envs.

## 7. Learning Progression (2M timestep quicktest, 2048 envs)

| Env Steps  | Reward  | Phase                          |
|-----------|---------|--------------------------------|
| 0         | -12.190 | Random — cube drops immediately|
| 327,680   | -6.817  | Less frequent dropping         |
| 655,360   | -3.049  | Basic grip forming             |
| 983,040   | +1.541  | Stable grip achieved           |
| 1,966,080 | +41.023 | Steady hold, minor adjustments |
| 3,276,800 | +83.209 | Consistent position control    |
| 4,915,200 | +103.363| Occasional orientation match   |
| 6,225,920 | +116.655| Stable hold, no rotation yet   |

### Observed behavior (from rollout video):
- Hand successfully grasps and holds cube
- Fingers make minor adjustments to maintain grip
- NO deliberate cube rotation observed at 6M steps
- Rotation is expected to emerge at 10-30M steps based on reward structure

## 8. Estimated Training Times (at ~11,500 steps/sec with 2048 envs)

| Total Steps | Estimated Time | Purpose                    |
|------------|----------------|----------------------------|
| 2M         | ~10 min        | Quick test (done ✓)        |
| 20M        | ~30 min        | Rotation emergence test    |
| 50M        | ~1.2 hours     | SAC training budget        |
| 100M       | ~2.4 hours     | Full PPO baseline per seed |
| 200M       | ~4.8 hours     | Extended training if needed|

## 9. Open Questions for Full Experiments

1. At what step count does deliberate rotation first emerge?
2. Does domain randomization slow convergence but improve robustness?
3. Can SAC match PPO's sample efficiency on this contact-rich task?
4. Does the asymmetric observation (privileged_state for critic) measurably
   help compared to using `state` for both actor and critic?
5. Would MJX-Warp backend improve throughput due to dynamic contact handling?
