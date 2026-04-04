#!/bin/bash
# =============================================================================
# train_ppo_dr.sh
#
# PPO with Domain Randomization — LeapCubeReorient
# Runs 3 seeds sequentially (100M steps each)
#
# Domain randomization varies per episode:
#   - Fingertip friction (0.5–1.0)
#   - Cube mass (±20%) and center of mass (±5mm)
#   - Initial joint positions (±0.05 rad)
#   - Joint friction loss (0.5×–2.0×)
#   - Joint armature (1.0×–1.05×)
#   - Link masses (±10%)
#   - Actuator stiffness Kp (±20%)
#   - Joint damping Kd (±20%)
#
# Run inside tmux:
#   tmux new -s ppo_dr
#   source ~/leap_hand_rl/activate.sh
#   bash ../scripts/train_ppo_dr.sh
#
#   Detach: Ctrl+B, then D
#   Re-attach: tmux attach -t ppo_dr
#
# Estimated time: ~2.5 hours per seed × 3 = ~7.5 hours total
# =============================================================================

set -e

export PYTHONUNBUFFERED=1

ENV_NAME="LeapCubeReorient"
NUM_TIMESTEPS=100000000
NUM_ENVS=2048
SEEDS=(0 1 2)

echo "=============================================="
echo "  PPO + Domain Randomization — LeapCubeReorient"
echo "  Seeds: ${SEEDS[*]}"
echo "  Steps per seed: $NUM_TIMESTEPS"
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
    echo "  DR Training Seed $SEED / ${SEEDS[-1]}"
    echo "  Started at: $(date)"
    echo "=============================================="

    uv --no-config run python learning/train_jax_ppo.py \
        --env_name "$ENV_NAME" \
        --num_timesteps "$NUM_TIMESTEPS" \
        --num_envs "$NUM_ENVS" \
        --domain_randomization True \
        --use_tb True \
        --seed "$SEED" \
        --suffix "dr_seed${SEED}"

    echo ""
    echo "  DR Seed $SEED complete at: $(date)"
done

echo ""
echo "=============================================="
echo "  ALL DR RUNS COMPLETE!"
echo "  Finished at: $(date)"
echo "=============================================="
