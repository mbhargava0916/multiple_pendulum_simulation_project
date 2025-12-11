"""
main_mbd_rigid.py

Entry point for the ME751 final project rigid double pendulum simulation.

This script:
  1) Defines the system and simulation parameters
  2) Runs the dynamics simulation
  3) Produces:
       - Static trajectory plots (2D + 3D, trajectories only)
       - 9 kinematics plots: x,y,z; vx,vy,vz; ax,ay,az
       - 2D animation (pendulum + trajectories)
       - 3D animation (pendulum + trajectories)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from rigid_pendulum.params import default_system_params
from rigid_pendulum.integrator import simulate
from rigid_pendulum.visualization import (
    plot_tip_trajectory,        # 2D trajectories only (Tip 1 + Tip 2)
    plot_tip_y_vs_time,
    animate_pendulum,           # 2D animation: pendulum + trajectories
    animate_pendulum_3d,        # 3D animation: pendulum + trajectories
    plot_kinematics_xyz,        # 9 plots: pos/vel/acc in x,y,z
    plot_tip_trajectories_3d,   # 3D trajectories only (Tip 1 + Tip 2)
)


def main():
    # 1. Define system and simulation parameters
    p = default_system_params()

    # Initial conditions:
    # theta1 measured from vertical; pi/2 ≈ horizontal along +x
    # theta2 a small offset from first link
    q0 = np.array([np.pi / 2.0, 0.1])   # [theta1, theta2]
    qdot0 = np.array([0.0, 0.0])        # initial angular velocities

    # 2. Run simulation
    results = simulate(p, q0, qdot0)

    # 3. Output directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # --- Static trajectory-only plots (for report) ---

    

    # 3D: Tip 1 (orange) and Tip 2 (green) trajectories embedded in x–z plane
    plot_tip_trajectories_3d(
        results,
        p,
        savepath=results_dir / "tip_trajectories_3d.png",
    )

    # --- Other static plots ---

    # 9 plots: position, velocity, acceleration components (x,y,z)
    plot_kinematics_xyz(results, p, save_dir=results_dir)

    # --- Animations (pendulum + trajectories) ---

    # 2D animation in x–y plane (links + trails)
    animate_pendulum(
        results,
        p,
        savepath=results_dir / "double_pendulum_2d.mp4",
    )

    # 3D animation embedded in x–z plane (links + trails, vertical swing)
    animate_pendulum_3d(
        results,
        p,
        savepath=results_dir / "double_pendulum_3d.mp4",
    )

    # Show all figures interactively (comment out if running headless)
    plt.show()


if __name__ == "__main__":
    main()
