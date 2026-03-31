#!/bin/bash
# =============================================================================
# train_ppo_baseline.sh
#
# Full PPO Baseline Training — LeapCubeReorient
# Runs 3 seeds sequentially (100M steps each)
#
# IMPORTANT: Run this inside tmux!
#
#   tmux new -s ppo_baseline
#   source ~/leap_hand_rl/activate.sh
#   bash ../scripts/train_ppo_baseline.sh
#
#   Detach: Ctrl+B, then D
#   Re-attach: tmux attach -t ppo_baseline
#
# Estimated time: ~2.5 hours per seed × 3 = ~7.5 hours total
# =============================================================================

set -e

# --- Fix Python output buffering (critical when running long jobs) ---
export PYTHONUNBUFFERED=1

# --- Configuration ---
ENV_NAME="LeapCubeReorient"
NUM_TIMESTEPS=100000000
NUM_ENVS=2048
SEEDS=(0 1 2)

echo "=============================================="
echo "  PPO Baseline Training — LeapCubeReorient"
echo "  Seeds: ${SEEDS[*]}"
echo "  Steps per seed: $NUM_TIMESTEPS"
echo "  Num Envs: $NUM_ENVS"
echo "  Started at: $(date)"
echo "=============================================="

# --- Verify GPU ---
echo ""
echo "[Pre-check] Verifying GPU..."
uv --no-config run python -c "
import jax
assert jax.default_backend() == 'gpu', 'GPU not found!'
print(f'  GPU verified: {jax.devices()[0]}')
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
        --seed "$SEED" \
        --use_tb True \
        --suffix "baseline_seed${SEED}"

    echo ""
    echo "  Seed $SEED complete at: $(date)"
done

echo ""
echo "=============================================="
echo "  ALL BASELINE RUNS COMPLETE!"
echo "  Finished at: $(date)"
echo "=============================================="
echo ""
echo "  Logs saved in: ./logs/"
echo "  Look for directories named *baseline_seed0, *baseline_seed1, *baseline_seed2"
