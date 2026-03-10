"""
load_environment.py — Load the LeapCubeReorient environment and inspect it.

This script loads the LEAP Hand cube reorientation task from MuJoCo Playground.
The environment is a Markov Decision Process (MDP) defined by (S, A, T, R, γ).

What you learn from the output:
  - observation_size: dimensionality of the state space S
    - 'state' (57,): what the policy sees — joint angles, velocities, cube pose,
      target orientation, previous action
    - 'privileged_state' (128,): extra info available only in simulation
      (contact forces, friction, etc.) — used for sim-to-real tricks
  - action_size: dimensionality of the action space A
    - 16: one target joint position per motor (4 fingers × 4 joints)

First run downloads MuJoCo Menagerie assets (~30s extra).

Usage:
  cd ~/leap_hand_rl/mujoco_playground
  uv --no-config run python ../scripts/load_environment.py
"""

from mujoco_playground import manipulation, registry

# registry.load() fetches the environment by name from Playground's registry.
# It returns an MjxEnv object — the JAX-compatible environment.
env = registry.load("LeapCubeReorient")

print("=" * 50)
print("  LeapCubeReorient Environment")
print("=" * 50)
print(f"  Observation size : {env.observation_size}")
print(f"  Action size      : {env.action_size}")
print(f"  Sim timestep (dt): {env.dt}")
print(f"  Substeps         : {env.n_substeps}")
print()

# List all available methods — useful for evaluation later
attrs = [a for a in dir(env) if not a.startswith("_")]
print("  Available attributes and methods:")
for a in attrs:
    print(f"    {a}")