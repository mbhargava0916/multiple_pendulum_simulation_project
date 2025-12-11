"""
sanity_check.py

Visual sanity check of double pendulum behavior using PyBullet
with a custom URDF created on the fly.

We:
  - write a simple double pendulum URDF into the current folder
    as 'double_pendulum_custom.urdf' (if it doesn't exist),
  - open the PyBullet GUI,
  - load that URDF,
  - set gravity and initial angles:
        theta1 = pi/2 (roughly horizontal)
        theta2 = 0.1  (small offset),
  - run the simulation so you can visually compare it
    with your Matplotlib double pendulum.
"""

import os
import time
import math
import textwrap

import pybullet as p


URDF_FILENAME = "double_pendulum_custom.urdf"

SIM_TIME = 10.0            # seconds
TIME_STEP = 1.0 / 240.0    # PyBullet default


def write_double_pendulum_urdf_if_needed():
    """Write a minimal double-pendulum URDF into this directory, if missing."""
    if os.path.exists(URDF_FILENAME):
        return

    # Simple planar double pendulum:
    # - gravity acts in -y
    # - links are boxes aligned along -y
    # - joints rotate about z-axis
    # link length = 1.0
    urdf_text = textwrap.dedent(
        """
        <?xml version="1.0"?>
        <robot name="double_pendulum_custom">
          <link name="base"/>

          <link name="link1">
            <inertial>
              <origin xyz="0 -0.5 0" rpy="0 0 0"/>
              <mass value="1.0"/>
              <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
            </inertial>
            <visual>
              <origin xyz="0 -0.5 0" rpy="0 0 0"/>
              <geometry>
                <box size="0.05 1.0 0.05"/>
              </geometry>
              <material name="blue">
                <color rgba="0 0 1 1"/>
              </material>
            </visual>
            <collision>
              <origin xyz="0 -0.5 0" rpy="0 0 0"/>
              <geometry>
                <box size="0.05 1.0 0.05"/>
              </geometry>
            </collision>
          </link>

          <link name="link2">
            <inertial>
              <origin xyz="0 -0.5 0" rpy="0 0 0"/>
              <mass value="1.0"/>
              <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
            </inertial>
            <visual>
              <origin xyz="0 -0.5 0" rpy="0 0 0"/>
              <geometry>
                <box size="0.05 1.0 0.05"/>
              </geometry>
              <material name="green">
                <color rgba="0 1 0 1"/>
              </material>
            </visual>
            <collision>
              <origin xyz="0 -0.5 0" rpy="0 0 0"/>
              <geometry>
                <box size="0.05 1.0 0.05"/>
              </geometry>
            </collision>
          </link>

          <!-- Joint 1: base -> link1, revolute about z -->
          <joint name="joint1" type="revolute">
            <parent link="base"/>
            <child link="link1"/>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <axis xyz="0 0 1"/>
            <limit lower="-3.14" upper="3.14" effort="0" velocity="0"/>
          </joint>

          <!-- Joint 2: link1 -> link2, revolute about z at end of link1 -->
          <joint name="joint2" type="revolute">
            <parent link="link1"/>
            <child link="link2"/>
            <origin xyz="0 -1.0 0" rpy="0 0 0"/>
            <axis xyz="0 0 1"/>
            <limit lower="-3.14" upper="3.14" effort="0" velocity="0"/>
          </joint>
        </robot>
        """
    ).strip()

    with open(URDF_FILENAME, "w", encoding="utf-8") as f:
        f.write(urdf_text)

    print(f"Created custom URDF: {URDF_FILENAME}")


def main():
    write_double_pendulum_urdf_if_needed()

    # 1. Connect to PyBullet GUI
    physics_client = p.connect(p.GUI)

    # 2. Set gravity: downward in -y
    p.setGravity(0.0, -9.81, 0.0)

    # 3. Load our custom double pendulum URDF
    pendulum_id = p.loadURDF(URDF_FILENAME, basePosition=[0, 0, 0])

    print("Loaded custom double pendulum with body ID:", pendulum_id)

    # 4. Set initial joint states comparable to your analytical model
    theta1_init = math.pi / 2.0   # ~ horizontal first link
    theta2_init = 0.1             # small offset for second link

    # joint indices 0 and 1 correspond to joint1 and joint2 in URDF
    p.resetJointState(pendulum_id, 0, targetValue=theta1_init, targetVelocity=0.0)
    p.resetJointState(pendulum_id, 1, targetValue=theta2_init, targetVelocity=0.0)

    # 5. Simulation loop
    p.setTimeStep(TIME_STEP)
    num_steps = int(SIM_TIME / TIME_STEP)
    print(f"Running simulation for {SIM_TIME} s "
          f"({num_steps} steps at dt={TIME_STEP})...")

    for i in range(num_steps):
        p.stepSimulation()
        time.sleep(TIME_STEP)  # approx real time

    print("Simulation finished. Close the GUI window to exit.")
    while p.isConnected():
        time.sleep(0.1)

    p.disconnect()


if __name__ == "__main__":
    main()
