#!/bin/bash
# =============================================================================
# train_sac_full.sh
#
# Full SAC Training — LeapCubeReorient
# Runs 3 seeds sequentially (50M steps each)
#
# Run inside tmux:
#   tmux new -s sac_training
#   source ~/leap_hand_rl/activate.sh
#   bash ../scripts/train_sac_full.sh
#
#   Detach: Ctrl+B, then D
#   Re-attach: tmux attach -t sac_training
#
# Estimated time: ~3 hours per seed × 3 = ~9 hours total
# =============================================================================

set -e

export PYTHONUNBUFFERED=1

ENV_NAME="LeapCubeReorient"
TOTAL_TIMESTEPS=50000000
NUM_ENVS=256
SEEDS=(0 1 2)

echo "=============================================="
echo "  SAC Training — LeapCubeReorient"
echo "  Seeds: ${SEEDS[*]}"
echo "  Steps per seed: $TOTAL_TIMESTEPS"
echo "  Num Envs: $NUM_ENVS"
echo "  Started at: $(date)"
echo "=============================================="

echo ""
echo "[Pre-check] Verifying GPU..."
uv --no-config run python -c "
import jax
assert jax.default_backend() == 'gpu', 'GPU not found!'
print(f'  GPU verified: {jax.devices()[0]}')
"

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "=============================================="
    echo "  SAC Training Seed $SEED / ${SEEDS[-1]}"
    echo "  Started at: $(date)"
    echo "=============================================="

    uv --no-config run python ../scripts/train_sac.py \
        --seed "$SEED" \
        --total_timesteps "$TOTAL_TIMESTEPS" \
        --num_envs "$NUM_ENVS" \
        --use_tb \
        --run_name "sac_${ENV_NAME}_baseline_seed${SEED}"

    echo ""
    echo "  SAC Seed $SEED complete at: $(date)"
done

echo ""
echo "=============================================="
echo "  ALL SAC RUNS COMPLETE!"
echo "  Finished at: $(date)"
echo "=============================================="
