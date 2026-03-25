#!/bin/bash
# Activate the LEAP Hand RL project environment
# Run with: source ~/leap_hand_rl/activate.sh

# Set JAX precision (only affects this terminal session)
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_FLAGS="--xla_gpu_triton_gemm_any=true"

# Move into the project
cd ~/leap_hand_rl/mujoco_playground

echo "==================================="
echo "  LEAP Hand RL Project Activated"
echo "  JAX precision: $JAX_DEFAULT_MATMUL_PRECISION"
echo "  XLA flags: $XLA_FLAGS"
echo "  Directory: $(pwd)"
echo "  Run scripts: uv --no-config run python ../scripts/SCRIPT.py"
echo "==================================="
