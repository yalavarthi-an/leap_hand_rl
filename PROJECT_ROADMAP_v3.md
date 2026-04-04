# Project Roadmap: PPO vs SAC on LEAP Hand

## CS 5180: Reinforcement Learning, Spring 2026
## Author: Anish Yalavarthi
## Last Updated: April 4, 2026

---

## PHASE 1: SETUP & VERIFICATION ✅ Complete

- [x] Install uv, clone MuJoCo Playground, install dependencies
- [x] Set up Git repo, activate.sh, project structure
- [x] Verify JAX GPU, load environment, run MDP transitions
- [x] Benchmark parallel envs (7,176 raw sps, ~11,500 training sps)
- [x] View LEAP Hand in interactive 3D viewer
- [x] Install ffmpeg, tensorboardX, tensorboard

## PHASE 2: PPO BASELINE ✅ Complete

- [x] Study train_jax_ppo.py line-by-line
- [x] Understand Brax PPO, JIT-compiled training, asymmetric observations
- [x] Quick tests: 1024/2048/4096 envs (2M steps each)
- [x] Verify rotation emergence at 30M steps
- [x] Full baseline: 100M steps × 3 seeds (seed 2 reached reward 230)
- [x] Rollout videos saved for all 3 seeds
- [ ] **TODO:** Re-run baseline with --use_tb True for TensorBoard data

## PHASE 3: PPO ABLATIONS — Partially Complete

### 3.1 Domain Randomization ✅ Complete
- [x] 3 seeds, 100M steps each, with TensorBoard logging
- [x] Result: DR reward ~148 vs baseline ~230 (36% lower)
- [x] DR converges earlier (~40M) but to lower ceiling
- [x] Very low seed variance (±2)

### 3.2 Num Envs Ablation ⬜ (have quicktest data, full runs optional)
- [x] Quicktest data: 512/1024/2048/4096 — minimal wall-clock difference
- [ ] Full 100M step runs (optional — quicktest finding is already clear)

## PHASE 4: SAC IMPLEMENTATION ⬜ Next Up

### 4.1 Study SAC Algorithm
- [ ] Understand SAC theory (max entropy, twin critics, replay buffer)
- [ ] Study how to interface off-policy algorithm with MJX environments
- [ ] Design SAC architecture for LeapCubeReorient

### 4.2 Implement SAC in JAX/Flax
- [ ] Actor network (Gaussian + tanh squashing)
- [ ] Twin Q-Critics (state+action input)
- [ ] GPU-resident replay buffer
- [ ] Automatic entropy tuning (alpha)
- [ ] Target network soft updates
- [ ] MJX interface via jax.vmap

### 4.3 Train and Evaluate SAC
- [ ] Quick test: 2M steps
- [ ] Full training: 50M steps × 3 seeds
- [ ] Hyperparameter sweep: LR, buffer size

## PHASE 5: EVALUATION & PLOTTING ⬜

- [ ] Re-run baseline with TensorBoard
- [ ] Generate all comparison plots
- [ ] Statistical analysis (mean ± std across seeds)
- [ ] Success rate metric

## PHASE 6: PAPER WRITING ⬜

- [ ] AAAI format report

---

## KEY FINDINGS SO FAR

| # | Finding | Source |
|---|---------|--------|
| 1 | Triton GEMM flag: 77% throughput boost on Blackwell | Phase 1 benchmark |
| 2 | Contact buffer (naconmax) dominates VRAM, not env count | Phase 1 scaling test |
| 3 | 1024-4096 envs: nearly identical wall-clock training time | Phase 1 scaling test |
| 4 | Rotation emerges at ~15-16M steps (reward crosses 150) | Phase 2 baseline |
| 5 | PPO baseline reaches reward 230 at 100M steps (still improving) | Phase 2 seed 2 |
| 6 | Domain randomization reduces sim reward by 36% | Phase 3.1 ablation |
| 7 | DR training time unaffected; DR converges earlier but lower | Phase 3.1 ablation |
| 8 | DR seed variance very low (±2) vs baseline (unknown) | Phase 3.1 ablation |
| 9 | Asymmetric obs (privileged_state for critic) used by default | Config analysis |
| 10 | Training throughput ~21,500 sps in JIT-compiled loop | Phase 2 timing |
