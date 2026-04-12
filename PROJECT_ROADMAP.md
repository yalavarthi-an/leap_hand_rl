# Project Roadmap: PPO vs SAC on LEAP Hand

## CS 5180: Reinforcement Learning, Spring 2026
## Author: Anish Yalavarthi
## Last Updated: April 12, 2026

---

## PHASE 1: SETUP & VERIFICATION ✅

- [x] uv, MuJoCo Playground, JAX+CUDA installed
- [x] Git repo, activate.sh, project structure
- [x] GPU verified, environment loaded, MDP transitions tested
- [x] Parallel benchmarks, interactive 3D viewer, ffmpeg, tensorboard

## PHASE 2: PPO BASELINE ✅

- [x] train_jax_ppo.py studied line-by-line
- [x] Brax PPO architecture, JIT compilation, asymmetric obs understood
- [x] Quick tests: 1024/2048/4096 envs
- [x] Rotation verified at 30M steps
- [x] Full baseline: 100M × 3 seeds (seed 2: reward 230)
- [x] Baseline re-run with TensorBoard (3 seeds complete)

## PHASE 3: PPO ABLATIONS ✅

- [x] Domain randomization: 3 seeds, reward ~148, TensorBoard logged
- [x] Num envs scaling: quicktest data (minimal wall-clock difference)

## PHASE 4: SAC ✅ (Negative Result — Thoroughly Documented)

- [x] SAC v1: basic implementation → Q-collapse
- [x] SAC v2: + reward scaling, UTD, alpha clamp → Q-collapse
- [x] SAC v3: + obs norm, asymmetric critic, LayerNorm → Q-collapse
- [x] Validation: SAC works on CartpoleBalance (proves code is correct)
- [x] Root cause: contact-rich reward discontinuity breaks Q-learning

## PHASE 5: AWAITING PROFESSOR GUIDANCE ⬜

### Option A: Alternative Off-Policy Approach
- [ ] Brax built-in SAC (JIT-compiled, on-device buffer)
- [ ] TD3 (no entropy framework)
- [ ] RLPD-style 10-critic ensemble
- [ ] Professor's recommendation

### Option B: PPO Across Multiple Environments
- [ ] Locomotion tasks (Go1, G1, Humanoid)
- [ ] Other manipulation (PandaPickCube, ALOHA)
- [ ] Contact-rich vs contact-light comparison
- [ ] MJX-Warp vs MJX-JAX backend comparison

## PHASE 6: PAPER WRITING ⬜

- [ ] AAAI format report
- [ ] All comparison plots from TensorBoard
- [ ] Statistical analysis

---

## KEY RESULTS

| Experiment | Reward | Status |
|-----------|--------|--------|
| PPO Baseline (100M) | ~230 | ✅ Complete |
| PPO + Domain Rand. (100M) | ~148 | ✅ Complete |
| SAC on LeapCubeReorient | Collapse | ✅ Negative result |
| SAC on CartpoleBalance | Learning | ✅ Validates code |

## COMPUTE LOG

| Run | Seeds | Hours | GPU |
|-----|-------|-------|-----|
| PPO Baseline | 3 | 4.6h | RTX 5070 |
| PPO + DR | 3 | 4.6h | RTX 5070 |
| PPO Baseline v2 (TB) | 3 | 4.6h | RTX 5070 |
| SAC v1-v3 tests | multiple | ~3h | RTX 5070 |
| SAC CartpoleBalance | 1 | 0.5h | RTX 5070 |
| **Total** | | **~17h** | |
