# Project Roadmap: PPO vs SAC on LEAP Hand

## CS 5180: Reinforcement Learning, Spring 2026
## Author: Anish Yalavarthi
## Last Updated: March 26, 2026

---

## PHASE 1: SETUP & VERIFICATION ✅ (Complete)

### 1.1 Environment Setup ✅
- [x] Install uv package manager
- [x] Clone MuJoCo Playground from source
- [x] Install all dependencies (JAX+CUDA, MuJoCo, MJX, Brax, Flax)
- [x] Set up project Git repo with proper .gitignore
- [x] Create project activation script (activate.sh)
- [x] Install ffmpeg for video rendering

### 1.2 GPU & Environment Verification ✅
- [x] Verify JAX sees RTX 5070 GPU (Script 01)
- [x] Load LeapCubeReorient environment (Script 02)
- [x] Run single MDP transition: s → a → s' + r (Script 03)
- [x] Benchmark parallel environments with Triton flag (~7,176 steps/sec) (Script 04)
- [x] View LEAP Hand in interactive 3D viewer (Script 05)

---

## PHASE 2: PPO BASELINE (Weeks 1-2) — IN PROGRESS

### 2.1 Understand the PPO Training Pipeline ✅
- [x] Study train_jax_ppo.py — imports, flags, config system
- [x] Understand Brax PPO integration and JIT-compiled training loop
- [x] Understand environment wrapper (wrap_for_brax_training)
- [x] Study pre-tuned hyperparameters for LeapCubeReorient
- [x] Understand asymmetric observations (state vs privileged_state)
- [x] Set JAX_DEFAULT_MATMUL_PRECISION=highest for reproducibility
- [x] Understand post-training evaluation and video rendering pipeline

### 2.2 Quick Tests and GPU Benchmarking ✅
- [x] Quick test: 2M steps, 1024 envs → reward 114.1, VRAM 2.4 GB
- [x] Quick test: 2M steps, 2048 envs → reward 116.7, VRAM 2.5 GB
- [x] Quick test: 2M steps, 4096 envs → reward 114.9, VRAM 3.0 GB
- [x] Finding: Fixed contact buffer (naconmax) dominates VRAM
- [x] Finding: Training time nearly identical across 1024-4096 envs
- [x] Selected 2048 envs as sweet spot for full training
- [x] Saved findings to results/quicktest_findings.md

### 2.2b Rotation Emergence Verification ✅
- [x] 30M step run with 2048 envs
- [x] Confirmed: cube rotation observed (1 successful rotation)
- [x] Rotation emerges around 15-30M steps as predicted
- [x] Learned tmux for persistent training sessions

### 2.3 Full Baseline Training ⬜ (NEXT)
- [ ] Install tmux for persistent sessions
- [ ] Run: 100M steps × 3 seeds (seed 0, 1, 2) with 2048 envs
- [ ] Estimated time: ~2.5 hours/seed × 3 = ~7.5 hours total
- [ ] Monitor with TensorBoard
- [ ] Verify convergence (reward plateau, multiple rotations)
- [ ] Save rollout videos for all 3 seeds
- [ ] Record final metrics: reward, throughput, wall-clock time

---

## PHASE 3: PPO ABLATIONS (Week 3, partial) ⬜

### 3.1 Domain Randomization Ablation
- [ ] Train PPO WITH domain randomization: 3 seeds, 100M steps each
- [ ] Compare: DR vs no-DR learning curves
- [ ] Analyze: does DR hurt convergence speed but improve robustness?

### 3.2 Parallel Environment Count Ablation
- [ ] Train PPO with 512 envs (1 seed, 100M steps)
- [ ] Train PPO with 1024 envs (1 seed, 100M steps)
- [ ] Already have 2048 envs from baseline
- [ ] Train PPO with 4096 envs (1 seed, 100M steps)
- [ ] Compare: throughput and final performance
- [ ] Note: quicktests suggest minimal wall-clock difference (contact buffer bound)

---

## PHASE 4: SAC IMPLEMENTATION & TRAINING (Weeks 3-4) ⬜

### 4.1 Implement SAC for MJX
- [ ] Write SAC Actor network (Gaussian policy with tanh squashing)
- [ ] Write SAC Twin Critics (Q-networks taking state+action)
- [ ] Implement GPU-resident replay buffer (JAX arrays)
- [ ] Implement automatic entropy temperature tuning
- [ ] Implement target network soft updates
- [ ] Interface with MJX environment via jax.vmap

### 4.2 Train SAC
- [ ] Quick test: 2M steps, verify learning signal exists
- [ ] Debug if needed: check Q-values, entropy, gradients
- [ ] Full training: 50M steps, 3 seeds
- [ ] Monitor Q-values for divergence (common failure mode)

### 4.3 SAC Hyperparameter Sensitivity
- [ ] Learning rate sweep: 1e-4, 3e-4, 1e-3
- [ ] Replay buffer size: 500K, 1M, 2M

### 4.4 SAC Failure Analysis (if needed)
- [ ] Check for Q-value overestimation
- [ ] Check entropy temperature behavior
- [ ] Document failure modes (valid negative result for paper)

---

## PHASE 5: EVALUATION & COMPARISON (Week 4-5) ⬜

### 5.1 Evaluation Protocol
- [ ] Run 50+ deterministic eval episodes per algorithm
- [ ] Measure: consecutive successful reorientations
- [ ] Measure: average episode return
- [ ] Generate evaluation videos

### 5.2 Generate Comparison Plots
- [ ] Learning curves: reward vs env steps (PPO vs SAC, 3 seeds, std bands)
- [ ] Wall-clock curves: reward vs training time
- [ ] Sample efficiency: steps to threshold reward
- [ ] Ablation plots: domain randomization, num_envs
- [ ] Success rate bar chart

### 5.3 Statistical Analysis
- [ ] Mean ± std across seeds for all metrics
- [ ] Note significance where applicable

---

## PHASE 6: PAPER WRITING (Week 5) ⬜

### 6.1 Report Structure (AAAI Format)
- [ ] Abstract
- [ ] Introduction
- [ ] Related Work
- [ ] Background (MDP, PPO, SAC, MuJoCo MJX)
- [ ] Method
- [ ] Experiments
- [ ] Results
- [ ] Discussion
- [ ] Conclusion

---

## SCRIPTS INDEX

| Phase | # | Script | Status |
|-------|---|--------|--------|
| 1 | 01 | `verify_gpu.py` | ✅ |
| 1 | 02 | `load_environment.py` | ✅ |
| 1 | 03 | `single_env_step.py` | ✅ |
| 1 | 04 | `parallel_benchmark.py` | ✅ |
| 1 | 05 | `view_hand.py` | ✅ |
| 2 | 06 | `06_train_ppo_baseline.sh` | ⬜ Ready to run |
| 3 | 07 | `07_train_ppo_ablations.sh` | ⬜ |
| 4 | 08 | `08_train_sac.py` | ⬜ |
| 5 | 09 | `09_eval_policy.py` | ⬜ |
| 5 | 10 | `10_plot_results.py` | ⬜ |

---

## KEY EXPERIMENTAL FINDINGS (so far)

1. **Triton GEMM flag** provides ~77% throughput improvement on RTX 5070
2. **Contact buffer (naconmax: 245760)** dominates VRAM, not env count
3. **1024/2048/4096 envs** produce nearly identical training wall-clock time
4. **2048 envs** selected as sweet spot (slightly better early learning)
5. **Rotation emerges at ~15-30M steps** after initial grip learning phase
6. **Training throughput: ~11,500 steps/sec** during full training loop
7. **Asymmetric observations** used (state for actor, privileged_state for critic)

---

## ESTIMATED COMPUTE BUDGET

| Experiment | Runs | Steps Each | Est. Hours Each | Total Hours |
|-----------|------|-----------|----------------|-------------|
| PPO baseline | 3 seeds | 100M | ~2.5h | ~7.5h |
| PPO + DR | 3 seeds | 100M | ~2.5h | ~7.5h |
| PPO num_envs ablation | 4 configs | 100M | ~2.5h | ~10h |
| SAC full | 3 seeds | 50M | ~1.5h | ~4.5h |
| SAC HP sweep | 6 runs | 20M | ~0.6h | ~3.5h |
| **Total** | | | | **~33h** |

Note: Estimates based on ~11,500 steps/sec throughput on RTX 5070 Laptop.
