import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from rigid_pendulum.params import default_system_params
from rigid_pendulum.integrator import simulate
from rigid_pendulum.visualization import (
    plot_tip_trajectory,
    plot_tip_y_vs_time,
    animate_pendulum,
    animate_pendulum_3d,
    plot_kinematics_xyz,
)


def main():
    # 1. Define problem
    p = default_system_params()

    # Start link 1 roughly horizontal, link 2 slightly perturbed
    q0 = np.array([np.pi / 2.0, 0.1])   # [theta1, theta2] from vertical
    qdot0 = np.array([0.0, 0.0])

    # 2. Run simulation
    results = simulate(p, q0, qdot0)

    # 3. Post-process / visualize
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Combined tip-1 and tip-2 trajectory in x–y plane
    plot_tip_trajectory(results, p,
                        savepath=results_dir / "tip_trajectories_xy.png")

    plot_tip_y_vs_time(results, p,
                       savepath=results_dir / "tip2_y_vs_time.png")

    # 9 required plots: x,y,z; vx,vy,vz; ax,ay,az
    plot_kinematics_xyz(results, p, save_dir=results_dir)

    # (Optional) 2D animation
    animate_pendulum(results, p,
                     savepath=results_dir / "double_pendulum_2d.mp4")

    # Required: 3D animation with trails
    animate_pendulum_3d(results, p,
                        savepath=results_dir / "double_pendulum_3d.mp4")

    plt.show()


if __name__ == "__main__":
    main()
