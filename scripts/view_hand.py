"""
view_hand.py — Launch the interactive 3D MuJoCo viewer for the LEAP Hand.

Opens a 3D window where you can see and interact with the LEAP Hand
and the cube in the simulation environment.

Viewer Controls:
  Left-click + drag    = rotate camera
  Right-click + drag   = pan camera
  Scroll wheel         = zoom in/out
  Double-click a body  = select it (shows info in the bottom bar)
  Ctrl + right-click   = apply force to a body (try dragging the cube!)
  Esc                  = close viewer

What you're seeing:
  - The LEAP Hand: a 16-DOF robotic hand with 4 fingers
    (4 joints per finger: base rotation, proximal, middle, distal)
  - A small cube resting in the hand's palm
  - The physics simulation running in real-time

Concepts:
  mj_model: The MuJoCo "model" — a static description of the scene:
    all bodies, joints, actuators, geometries, materials, constraints.
    This is defined by an MJCF XML file and never changes during simulation.

  mj_data: The MuJoCo "data" — the current dynamic state of the simulation:
    joint positions/velocities, contact forces, sensor readings, etc.
    This changes every simulation step.

  Think of it as: mj_model = the blueprint, mj_data = the current snapshot.

Usage:
  cd ~/leap_hand_rl/mujoco_playground
  uv --no-config run python ../scripts/view_hand.py
"""

import mujoco
import mujoco.viewer
from mujoco_playground import registry

# Load the environment (this gives us access to the MuJoCo model)
env = registry.load("LeapCubeReorient")

# Get the MuJoCo model — the static scene description
# env.mj_model contains everything: the hand, the cube, the floor,
# joint limits, actuator gains, contact parameters, etc.
mj_model = env.mj_model

# Create MuJoCo data — the dynamic state container
# This allocates memory for joint positions, velocities, forces, etc.
mj_data = mujoco.MjData(mj_model)

# Reset to the initial configuration
# This places all joints at their default positions (as defined in the XML)
mujoco.mj_resetData(mj_model, mj_data)

print("=" * 50)
print("  LEAP Hand Interactive Viewer")
print("=" * 50)
print()
print("  Model info:")
print(f"    Bodies    : {mj_model.nbody}")
print(f"    Joints    : {mj_model.njnt}")
print(f"    Actuators : {mj_model.nu}")
print(f"    Geoms     : {mj_model.ngeom}")
print()
print("  Controls:")
print("    Left-click + drag    = rotate camera")
print("    Right-click + drag   = pan camera")
print("    Scroll wheel         = zoom in/out")
print("    Double-click a body  = select it")
print("    Ctrl + right-click   = apply force")
print("    Esc                  = close viewer")
print()
print("  Opening viewer...")

# Launch the interactive viewer
# This opens a new window and blocks until you close it (press Esc)
mujoco.viewer.launch(mj_model, mj_data)

print("  Viewer closed.")