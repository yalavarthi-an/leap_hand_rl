"""
verify_gpu.py — Verify that JAX detects the GPU.

This is the very first check you should run after installation.
JAX is Google's numerical computing library that runs on GPU via XLA compilation.
If this script prints 'cpu' instead of 'gpu', nothing else will work properly.

What this script checks:
  - JAX is installed correctly
  - JAX can find the NVIDIA GPU via CUDA drivers
  - The GPU device is accessible

Usage:
  cd ~/leap_hand_rl/mujoco_playground
  uv --no-config run python ../scripts/verify_gpu.py
"""

import jax

print("=" * 50)
print("  JAX GPU Verification")
print("=" * 50)
print(f"  JAX version : {jax.__version__}")
print(f"  Backend     : {jax.default_backend()}")
print(f"  Devices     : {jax.devices()}")
print(f"  GPU count   : {jax.device_count()}")
print("=" * 50)

if jax.default_backend() == "gpu":
    print("  ✓ SUCCESS — JAX is using your GPU!")
else:
    print("  ✗ PROBLEM — JAX is using CPU.")
    print("    Fix: run 'unset LD_LIBRARY_PATH' then try again.")
    print("    If still failing, reinstall JAX: pip install jax[cuda12]")