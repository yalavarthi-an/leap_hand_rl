#!/bin/bash
# =============================================================================
# 06_train_ppo_baseline.sh
# 
# Full PPO Baseline Training — LeapCubeReorient
# Runs 3 seeds sequentially (100M steps each)
# 
# IMPORTANT: Run this inside tmux so it survives terminal closure!
#
#   tmux new -s ppo_baseline
#   source ~/leap_hand_rl/activate.sh
#   bash ../scripts/06_train_ppo_baseline.sh
#
# To detach from tmux (training keeps running):
#   Press Ctrl+B, then D
#
# To re-attach and check progress:
#   tmux attach -t ppo_baseline
#
# Estimated time: ~2.5 hours per seed × 3 = ~7.5 hours total
# =============================================================================

set -e  # Exit on error

# --- Configuration ---
ENV_NAME="LeapCubeReorient"
NUM_TIMESTEPS=100000000   # 100M steps per seed
NUM_ENVS=2048             # Sweet spot for RTX 5070 (8 GB VRAM)
SEEDS=(0 1 2)
USE_TB="True"             # Enable TensorBoard logging
LOG_DIR="./logs"

echo "=============================================="
echo "  PPO Baseline Training — LeapCubeReorient"
echo "  Seeds: ${SEEDS[*]}"
echo "  Steps per seed: $NUM_TIMESTEPS"
echo "  Num Envs: $NUM_ENVS"
echo "  Started at: $(date)"
echo "=============================================="

# --- Verify GPU before starting ---
echo ""
echo "[Pre-check] Verifying GPU..."
uv --no-config run python -c "
import jax
assert jax.default_backend() == 'gpu', 'GPU not found!'
print(f'  GPU verified: {jax.devices()[0]}')
print(f'  JAX precision: highest')
"

# --- Train each seed ---
for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "=============================================="
    echo "  Training Seed $SEED / ${SEEDS[-1]}"
    echo "  Started at: $(date)"
    echo "=============================================="

    uv --no-config run python learning/train_jax_ppo.py \
        --env_name "$ENV_NAME" \
        --num_timesteps "$NUM_TIMESTEPS" \
        --num_envs "$NUM_ENVS" \
        --use_tb "$USE_TB" \
        --seed "$SEED" \
        --suffix "baseline_seed${SEED}" \
        2>&1 | tee "$LOG_DIR/ppo_baseline_seed${SEED}_terminal.log"

    echo ""
    echo "  Seed $SEED complete at: $(date)"
    echo "  Log saved to: $LOG_DIR/ppo_baseline_seed${SEED}_terminal.log"
done

echo ""
echo "=============================================="
echo "  ALL BASELINE RUNS COMPLETE!"
echo "  Finished at: $(date)"
echo "=============================================="
echo ""
echo "  Results saved in:"
for SEED in "${SEEDS[@]}"; do
    echo "    $LOG_DIR/*baseline_seed${SEED}/"
done
echo ""
echo "  To view TensorBoard:"
echo "    cd ~/leap_hand_rl/mujoco_playground"
echo "    uv --no-config run tensorboard --logdir=./logs --port 6006"
