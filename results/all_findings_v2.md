# Experimental Findings — All Phases

## CS 5180 Project: PPO vs SAC on LEAP Hand
## Author: Anish Yalavarthi
## Last Updated: April 4, 2026

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
- **Brax:** PPO implementation from brax.training.agents.ppo
- **Physics backend:** MJX-JAX (`--impl jax`)
- **Precision:** `JAX_DEFAULT_MATMUL_PRECISION=highest` (full float32)
- **XLA flags:** `--xla_gpu_triton_gemm_any=true` (Triton GEMM optimization)
- **Logging:** TensorBoardX for metric logging (installed after initial runs)

## 3. Environment Details

- **Environment:** LeapCubeReorient (MuJoCo Playground)
- **Observation space:** `state` (57 dims), `privileged_state` (128 dims)
- **Action space:** 16 continuous (joint target positions)
- **Sim timestep (dt):** 0.01s
- **Control timestep (ctrl_dt):** 0.05s
- **Episode length:** 1000 steps
- **Success threshold:** 0.1 radians
- **Contact buffer (naconmax):** 245,760 (fixed pre-allocation)

### Reward Structure
| Component       | Weight  | Description                              |
|----------------|---------|------------------------------------------|
| orientation    | +5.0    | Cube orientation close to target         |
| position       | +0.5    | Cube position stability                  |
| success_reward | +100.0  | Bonus when within 0.1 rad of target      |
| hand_pose      | -0.5    | Penalty for unnatural hand configuration |
| termination    | -100.0  | Penalty when cube is dropped             |
| action_rate    | -0.001  | Penalty for jerky actions                |
| energy         | -0.001  | Penalty for high energy usage            |

### Reward Level Interpretation
| Reward Range | Meaning                                        |
|-------------|------------------------------------------------|
| -12 to 0    | Random policy, cube drops frequently           |
| 0 to 115    | Stable grip, no deliberate rotation            |
| 115 to 150  | First rotations emerging (occasional success)  |
| 150 to 215  | ~1 successful reorientation per episode        |
| 215 to 315  | ~2 successful reorientations per episode       |
| 315+        | 3+ consecutive reorientations (skilled policy) |

## 4. PPO Configuration

### Pre-tuned by Playground developers (for A100 GPU)
| Parameter                    | Default (A100) | Our Setting (RTX 5070) |
|-----------------------------|----------------|------------------------|
| Network (actor & critic)    | (512, 256, 128)| (512, 256, 128)        |
| Actor obs key               | `state` (57d)  | `state` (57d)          |
| Critic obs key              | `privileged_state` (128d) | `privileged_state` (128d) |
| Learning rate               | 3e-4           | 3e-4                   |
| Discount (γ)                | 0.99           | 0.99                   |
| Clipping (ε)                | 0.2            | 0.2                    |
| Entropy cost                | 0.01           | 0.01                   |
| Unroll length               | 40             | 40                     |
| Num updates per batch       | 4              | 4                      |
| Num minibatches             | 32             | 32                     |
| Num envs                    | 8192           | **2048** (VRAM limited) |
| Num timesteps               | 200,000,000    | **100,000,000**        |
| Reward scaling              | 1.0            | 1.0                    |
| Observation normalization   | True           | True                   |

### Asymmetric Observation Design
- **Actor** uses `state` (57 dims): only information measurable on real hardware
  (joint angles, velocities, cube pose from vision, target orientation, prev action)
- **Critic** uses `privileged_state` (128 dims): simulation-only information
  (contact forces, friction coefficients, physics parameters)
- **Purpose:** Better advantage estimates during training while keeping the
  deployed policy compatible with real-robot sensor limitations

---

## 5. GPU Benchmark Results

### Raw Environment Throughput (jax.vmap benchmark)
| Metric                | Without Triton | With Triton Flag |
|-----------------------|---------------|------------------|
| Steps/second          | 4,051         | 7,176            |
| Per-step latency      | 252.8 ms      | 142.7 ms         |
| Speedup               | baseline      | **1.77×**        |

### Parallel Environment Scaling (2M step quicktests)
| Num Envs | VRAM Used | JIT Time | Train Time | Final Reward | Throughput |
|----------|-----------|----------|------------|-------------|------------|
| 1024     | ~2.4 GB   | 51.4s    | 535.6s     | 114.1       | ~11,600 sps|
| 2048     | ~2.5 GB   | 58.2s    | 551.1s     | 116.7       | ~11,300 sps|
| 4096     | ~3.0 GB   | 57.9s    | 545.1s     | 114.9       | ~11,400 sps|

### Key Finding: Fixed Contact Buffer Dominates VRAM
The `naconmax: 245760` parameter pre-allocates a fixed-size contact buffer
(~1.5 GB) regardless of num_envs due to MJX-JAX's SIMD constraint requiring
static array shapes at compile time.

- Doubling num_envs from 1024→4096 only increased VRAM by ~0.6 GB
- Training wall-clock time was nearly identical across all settings
- **Selected 2048 as optimal for RTX 5070** (slight early learning advantage)

---

## 6. PPO Baseline Results (100M steps, 2048 envs)

### Seed 2 Learning Progression (complete data)
| Env Steps    | Reward  | Phase                              |
|-------------|---------|-------------------------------------|
| 0           | -11.588 | Random — cube drops immediately     |
| 5,570,560   | 104.786 | Stable grip achieved                |
| 11,141,120  | 125.637 | Grip refinement                     |
| 16,711,680  | 156.048 | **First rotations emerging**        |
| 22,282,240  | 168.431 | Consistent single rotations         |
| 33,423,360  | 163.504 | Plateau with variance               |
| 44,564,480  | 170.845 | Steady improvement                  |
| 55,705,600  | 171.995 | Reliable rotation                   |
| 66,846,720  | 175.668 | Improving consistency               |
| 77,987,840  | 181.936 | Multiple rotation attempts          |
| 89,128,960  | 183.708 | Near convergence                    |
| 100,270,080 | 199.684 | Late improvement                    |
| 105,840,640 | **230.434** | **Multiple consecutive rotations**|

### Key Observations
- **Grip phase (0–5M):** Reward climbs from -12 to +105. Policy learns not to drop cube.
- **Rotation emergence (~15M):** Reward jumps past 150, indicating first success_reward bonuses.
- **Plateau (20M–90M):** Reward hovers around 165–185. Policy reliably achieves 1 rotation.
- **Late breakthrough (100M+):** Reward jumps to 230, suggesting 2+ rotations per episode.
- **Policy was still improving at 100M** — 200M steps (as Playground developers use) would yield higher rewards.

### Training Performance
- **JIT compilation:** 46.0 seconds
- **Training time:** 4,914 seconds (81.9 minutes) per seed
- **Effective throughput:** ~21,500 steps/sec (higher than raw benchmark due to JIT-compiled training loop)
- **Total time for 3 seeds:** ~4.6 hours

### Rollout Video Observations (Seed 2, 100M steps)
- Hand maintains stable grip on cube
- Fingers execute coordinated rotation movements
- At least 1 deliberate cube reorientation observed
- Policy shows smooth, non-jerky movements

---

## 7. Domain Randomization Ablation (100M steps, 2048 envs, 3 seeds)

### Randomized Parameters
| Parameter               | Range            | Per-Joint? | Real-World Motivation          |
|------------------------|------------------|------------|-------------------------------|
| Fingertip friction     | U(0.5, 1.0)     | Per finger | Wear, dust, moisture          |
| Cube inertia           | ×U(0.8, 1.2)    | Global     | Manufacturing variation       |
| Cube CoM position      | ±5mm             | Global     | Imperfect geometry            |
| Initial joint angles   | ±0.05 rad        | Per joint  | Calibration error             |
| Joint friction loss    | ×U(0.5, 2.0)    | Per joint  | Temperature, wear             |
| Joint armature         | ×U(1.0, 1.05)   | Per joint  | Motor inertia variation       |
| Link masses            | ×U(0.9, 1.1)    | Per body   | CAD model approximation       |
| Actuator stiffness (Kp)| ×U(0.8, 1.2)    | Per actuator| Motor gain variation         |
| Joint damping (Kd)     | ×U(0.8, 1.2)    | Per joint  | Lubricant, temperature        |

### DR Results (from TensorBoard)
| Seed | Final Reward (smoothed) | Training Time |
|------|------------------------|---------------|
| 0    | ~148                   | ~1.4 hours    |
| 1    | ~145                   | ~1.4 hours    |
| 2    | ~147                   | ~1.4 hours    |
| **Mean ± Std** | **~147 ± 2** | |

### DR Learning Progression (from TensorBoard, approximate)
| Env Steps  | Approx Reward | Phase                          |
|-----------|---------------|--------------------------------|
| 0         | ~-12          | Random                         |
| 5M        | ~60           | Learning grip (slower than baseline) |
| 10M       | ~100          | Basic holding                  |
| 20M       | ~130          | Approaching plateau            |
| 40M       | ~140          | Near convergence               |
| 60M       | ~145          | Plateau                        |
| 100M      | ~148          | Converged                      |

### Baseline vs Domain Randomization Comparison
| Metric                    | Baseline (no DR) | With DR     | Difference |
|--------------------------|------------------|-------------|------------|
| Final reward (100M)      | ~230 (seed 2)   | ~148 (mean) | **-36%**   |
| Training time per seed   | ~82 min          | ~84 min     | ~Same      |
| Reward at 20M steps      | ~168             | ~130        | -23%       |
| Convergence point        | Still improving  | ~40-50M     | DR converges earlier |
| Seed variance            | N/A (1 seed)     | Low (±2)    | Very consistent |

### Key Findings — Domain Randomization
1. **DR reduces simulation reward by ~36%** but produces policies robust to
   physics parameter variation across all 9 randomized dimensions.
2. **Training wall-clock time is unaffected** by DR — the randomization adds
   negligible computational overhead.
3. **DR policies converge earlier** (~40-50M steps) compared to baseline
   (still improving at 100M). The robust policy has a lower but more stable ceiling.
4. **Very low variance across seeds** (±2 reward) suggests DR training is
   highly reproducible — the randomization itself provides exploration.
5. **DR policies plateau lower** because the evaluation also uses randomized
   physics. In fixed physics, DR policies would score higher (~180-200 estimated).

---

## 8. Lessons Learned (Practical Notes)

### Infrastructure
- **Always use tmux** for training runs >10 minutes to prevent loss from terminal closure
- **PYTHONUNBUFFERED=1** is essential when piping Python output through tee or logging
- **TensorBoardX** must be installed separately from TensorBoard (writer vs viewer)
- **activate.sh** pattern keeps project env vars isolated from other workflows (Isaac Sim, conda)

### GPU Optimization
- **Triton GEMM flag** provides 77% throughput improvement on Blackwell GPUs — always enable
- **XLA_PYTHON_CLIENT_PREALLOCATE=false** prevents JAX from grabbing all VRAM at startup
- **Close browser tabs** during training — Chrome RAM pressure causes swap, slowing GPU
- **Charger required** — RTX 5070 draws 49W at full load, training on battery is impractical

### Training Monitoring
- **20 evals over 100M steps** = 1 printout every 5M steps (~7 min intervals)
- **JIT compilation** takes 45-58 seconds — first printout appears ~8 min after launch
- **Rollout videos** require ffmpeg: `sudo apt install ffmpeg`
- **Baseline terminal output was lost** for seeds 0/1 due to no persistent logging — always use --use_tb True

---

## 9. Remaining Experiments

### To Re-Run (with TensorBoard logging)
- [ ] PPO Baseline: 3 seeds, 100M steps, --use_tb True
  (Need TensorBoard data for all seeds for proper learning curve comparison)

### Phase 3.2: Num Envs Ablation
- [ ] 512, 1024, 2048, 4096 envs — 1 seed each, 100M steps
  (Quicktest data suggests minimal wall-clock difference; full runs will confirm)

### Phase 4: SAC Implementation
- [ ] Implement SAC in JAX/Flax for MJX environments
- [ ] Train SAC: 50M steps, 3 seeds
- [ ] Hyperparameter sensitivity: learning rate, buffer size

### Phase 5: Evaluation & Plotting
- [ ] Generate comparison plots from TensorBoard data
- [ ] Statistical analysis across seeds
- [ ] Success rate metric (consecutive reorientations)

---

## 10. Open Research Questions

1. ~~At what step count does deliberate rotation first emerge?~~
   **ANSWERED: ~15-16M steps** (reward crosses 150, indicating success_reward bonuses)

2. ~~Does domain randomization slow convergence but improve robustness?~~
   **ANSWERED: Yes.** DR reduces final sim reward by ~36% but converges earlier
   and produces robust policies with very low seed variance.

3. Can SAC match PPO's sample efficiency on this contact-rich task?
   **PENDING — Phase 4**

4. Does the asymmetric observation (privileged_state for critic) measurably
   help compared to using `state` for both actor and critic?
   **PENDING — possible additional ablation**

5. Would MJX-Warp backend improve throughput due to dynamic contact handling?
   **PENDING — possible bonus experiment**
