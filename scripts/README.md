# Scripts

All experiment scripts, numbered in execution order.

## How to run

Every script runs from inside the `mujoco_playground/` directory:

```bash
cd ~/leap_hand_rl/mujoco_playground
uv --no-config run python ../scripts/SCRIPT_NAME.py
```

## Script Index

### Setup & Verification (Step 1-2)
| # | Script | Purpose |
|---|--------|---------|
| 01 | `verify_gpu.py` | Check JAX sees the GPU |
| 02 | `load_environment.py` | Load LeapCubeReorient, inspect obs/action spaces |
| 03 | `parallel_benchmark.py` | Benchmark 1024 parallel envs on GPU |
| 04 | `view_hand.py` | Launch interactive 3D viewer |

### Training (coming next)
| # | Script | Purpose |
|---|--------|---------|
| 05 | `train_ppo_baseline.py` | PPO baseline (3 seeds) |
| 06 | `train_ppo_ablations.py` | PPO ablations (DR, num_envs) |
| 07 | `train_sac.py` | SAC implementation for MJX |

### Analysis (coming later)
| # | Script | Purpose |
|---|--------|---------|
| 08 | `eval_policy.py` | Evaluate trained policies |
| 09 | `plot_results.py` | Generate comparison figures |