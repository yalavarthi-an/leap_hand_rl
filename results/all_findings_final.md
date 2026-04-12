# Complete Experimental Findings

## CS 5180 Project: PPO vs SAC on LEAP Hand
## Author: Anish Yalavarthi
## Last Updated: April 12, 2026

---

## 1. Hardware & Software

- **GPU:** NVIDIA RTX 5070 Laptop (8,151 MiB VRAM), Blackwell, CUDA 13.1
- **System:** Ubuntu 24.04, Python 3.12.12, JAX 0.6.2
- **Framework:** MuJoCo Playground (MJX-JAX), Brax PPO
- **Precision:** float32 (JAX_DEFAULT_MATMUL_PRECISION=highest)
- **Optimization:** XLA Triton GEMM enabled (+77% throughput)

## 2. Environment: LeapCubeReorient

- Observation: `state` (57d), `privileged_state` (128d)
- Action: 16 continuous (joint target positions)
- Episode: 1000 steps, success threshold 0.1 rad
- Contact buffer: naconmax=245,760 (fixed ~1.5 GB allocation)

## 3. PPO Baseline (100M steps, 2048 envs, 3 seeds)

### Pre-tuned Configuration (from Playground developers)
- Network: (512, 256, 128) tapered MLP, actor+critic
- Asymmetric: actor sees `state`, critic sees `privileged_state`
- LR: 3e-4, γ: 0.99, ε: 0.2, entropy: 0.01
- Unroll: 40 steps, 4 epochs, 32 minibatches

### Results (Seed 2 — complete data)
| Steps | Reward | Phase |
|-------|--------|-------|
| 0 | -11.6 | Random |
| 5.6M | 104.8 | Grip learned |
| 16.7M | 156.0 | First rotations |
| 44.6M | 170.8 | Consistent rotation |
| 100.3M | 199.7 | Multiple rotations |
| 105.8M | 230.4 | Best performance |

- Training time: 82 min/seed, throughput ~21,500 sps
- Rotation emerges at ~15-16M steps
- Still improving at 100M (200M recommended for full convergence)

## 4. Domain Randomization Ablation (100M steps, 3 seeds)

### 9 Randomized Parameters
Fingertip friction (0.5-1.0), cube mass (±20%), cube CoM (±5mm),
initial joints (±0.05 rad), joint friction (0.5×-2.0×),
armature (1.0×-1.05×), link masses (±10%), Kp (±20%), Kd (±20%)

### Results
| Metric | Baseline | With DR |
|--------|----------|---------|
| Final reward | ~230 | ~148 (±2) |
| Convergence | Still improving at 100M | Converged by ~40M |
| Training time | ~82 min/seed | ~84 min/seed |
| Seed variance | N/A | Very low (±2) |

**Finding:** DR reduces sim reward by 36% but produces robust policies.
Training time unaffected. DR converges earlier to a lower ceiling.

## 5. GPU Scaling (Quick Tests — 2M steps each)

| Num Envs | VRAM | Train Time | Reward | Throughput |
|----------|------|------------|--------|------------|
| 1024 | ~2.4 GB | 536s | 114.1 | ~11,600 sps |
| 2048 | ~2.5 GB | 551s | 116.7 | ~11,300 sps |
| 4096 | ~3.0 GB | 545s | 114.9 | ~11,400 sps |

**Finding:** Fixed contact buffer (naconmax) dominates VRAM. Doubling
envs from 1024→4096 adds only 0.6 GB. Training time nearly identical.

## 6. SAC Implementation and Failure Analysis

### Three Versions Attempted

**v1 (Basic SAC):**
- Actor/Critic both see `state` (57d)
- No obs normalization, no layer norm
- Result: Q-values frozen at -5.0, α collapsed to 0.0001

**v2 (+ Stability Fixes):**
- Added: reward scaling (×0.1), UTD ratio=4, α clamped at 0.01
- Result: Q-values frozen at -0.50 (= -5.0 × 0.1), critic loss→0.000

**v3 (Full Rewrite):**
- Added: observation normalization (Welford's algorithm)
- Added: asymmetric actor-critic (critic sees privileged_state 128d)
- Added: layer normalization on Q-networks (per RLPD)
- Added: 3-layer critic (256, 256, 256)
- Result: Q-values initially move (1.50 → -0.51), then freeze. Same collapse.

### Root Cause Analysis
All versions exhibit identical failure pattern:
1. Q-networks converge to predicting a constant value
2. Critic loss drops to 0.000 (both Q-nets predict same constant)
3. Actor receives zero gradient signal (all actions evaluated equally)
4. Policy stops improving, reward stuck at -5.05 (cube drops every episode)

**Hypothesized cause:** The termination penalty (-100) creates a sharp
discontinuity in the Q-function that neural network approximators cannot
model accurately. The (1-done) term in the Bellman target creates a cliff
between "holding" (Q ≈ positive) and "dropping" (Q ≈ -100 × reward_scale).
Q-networks smooth over this cliff, leading to systematic approximation
errors that cascade into collapse.

### Validation: SAC Works on CartpoleBalance
Same SAC algorithm tested on CartpoleBalance (5d obs, 1d action):
- Q1: -0.45 → +2.71 (steadily rising — healthy)
- α: 0.86 → 0.14 (natural decrease — auto-tuning working)
- Critic loss: 0.021 → 0.003 (meaningful, not zero)
- Reward: positive and improving

**Conclusion:** SAC implementation is correct. Failure is specific to
contact-rich dexterous manipulation with large termination penalties.

## 7. Key Findings Summary

| # | Finding |
|---|---------|
| 1 | PPO successfully learns cube reorientation (reward 230 at 100M steps) |
| 2 | Rotation emerges at ~15-16M steps after initial grip learning phase |
| 3 | Domain randomization reduces sim reward by 36% but increases robustness |
| 4 | DR training time identical to baseline; DR converges earlier |
| 5 | Triton GEMM flag gives 77% throughput boost on Blackwell GPUs |
| 6 | Contact buffer (naconmax) dominates VRAM, not environment count |
| 7 | SAC exhibits Q-value collapse on LeapCubeReorient (3 versions tried) |
| 8 | SAC works correctly on CartpoleBalance (validated implementation) |
| 9 | On-policy PPO is more suitable than off-policy SAC for contact-rich manipulation |
| 10 | Asymmetric observations (privileged_state for critic) used by PPO |

## 8. Open Questions

1. Would Brax's built-in JIT-compiled SAC behave differently?
2. Could TD3 succeed where SAC fails (no entropy framework)?
3. Would RLPD-style 10-critic ensemble prevent Q-collapse?
4. Does PPO's advantage estimation fundamentally handle reward discontinuities better?
5. Would reshaping the reward (removing termination penalty) help SAC converge?
