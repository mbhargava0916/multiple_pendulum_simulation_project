"""
triple_pendulum_sim.py

Simple planar triple pendulum simulation and plotting.

Model:
- 3 point masses m1, m2, m3
- 3 massless rods of lengths l1, l2, l3
- Gravity acts in -y
- Generalized coordinates: theta1, theta2, theta3 (angles from vertical)

Equations of motion are taken from a standard Lagrangian formulation
for a triple pendulum with point masses at the rod ends. They are
implemented as a 3x3 linear system in the angular accelerations
(theta1_ddot, theta2_ddot, theta3_ddot). See e.g. Yesilyurt (2019),
"Equations of Motion Formulation of a Pendulum Containing N-point Masses".
"""

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


@dataclass
class TriplePendulumParams:
    g: float = 9.81
    m1: float = 1.0
    m2: float = 1.0
    m3: float = 1.0
    l1: float = 1.0
    l2: float = 1.0
    l3: float = 1.0
    dt: float = 0.001
    t_final: float = 10.0

    # Initial conditions (angles from vertical, rad; angular velocities rad/s)
    theta1_0: float = np.pi / 2.0
    theta2_0: float = 0.1
    theta3_0: float = -0.1
    omega1_0: float = 0.0
    omega2_0: float = 0.0
    omega3_0: float = 0.0


def triple_pendulum_rhs(t, y, p: TriplePendulumParams):
    """
    Right-hand side of the triple pendulum ODE.

    y = [theta1, theta2, theta3, omega1, omega2, omega3]
    Returns dy/dt in the same order.
    """
    theta1, theta2, theta3, omega1, omega2, omega3 = y
    g = p.g
    m1, m2, m3 = p.m1, p.m2, p.m3
    l1, l2, l3 = p.l1, p.l2, p.l3

    # Precompute sines and cosines of angle differences
    s12 = np.sin(theta1 - theta2)
    c12 = np.cos(theta1 - theta2)
    s13 = np.sin(theta1 - theta3)
    c13 = np.cos(theta1 - theta3)
    s21 = np.sin(theta2 - theta1)  # = -s12
    c21 = c12
    s23 = np.sin(theta2 - theta3)
    c23 = np.cos(theta2 - theta3)
    s31 = np.sin(theta3 - theta1)  # = -s13
    c31 = c13
    s32 = np.sin(theta3 - theta2)  # = -s23
    c32 = c23

    # ------------------------------
    # Build 3x3 mass matrix A and RHS b such that A * alpha = b,
    # where alpha = [theta1_ddot, theta2_ddot, theta3_ddot].
    # Equations based on Yesilyurt 2019 (triple pendulum EOM).
    # ------------------------------

    # --- Coefficients for accelerations (A matrix) ---

    # Equation for theta1 (eq. (30) style)
    A11 = (m1 + m2 + m3) * l1 ** 2
    A12 = (m2 + m3) * l1 * l2 * c12
    A13 = m3 * l1 * l3 * c13

    # Equation for theta2 (eq. (31))
    A21 = (m2 + m3) * l1 * l2 * c21
    A22 = (m2 + m3) * l2 ** 2
    A23 = m3 * l2 * l3 * c23

    # Equation for theta3 (eq. (32))
    A31 = m3 * l1 * l3 * c13
    A32 = m3 * l2 * l3 * c23
    A33 = m3 * l3 ** 2

    A = np.array([
        [A11, A12, A13],
        [A21, A22, A23],
        [A31, A32, A33],
    ])

    # --- Non-acceleration terms (N vector), so that A*alpha + N = 0 => A*alpha = -N ---

    # Equation 1 (theta1)
    N1 = (
        g * l1 * (m1 + m2 + m3) * np.sin(theta1)
        + (m2 + m3) * l1 * l2 * s12 * omega1 * omega2
        + m3 * l1 * l3 * s13 * omega1 * omega3
        + (m2 + m3) * l1 * l2 * s21 * (omega1 - omega2) * omega2
        + m3 * l1 * l3 * s31 * (omega1 - omega3) * omega3
    )

    # Equation 2 (theta2)
    N2 = (
        g * l2 * (m2 + m3) * np.sin(theta2)
        + (m2 + m3) * l1 * l2 * s21 * omega1 * omega2
        + m3 * l2 * l3 * s23 * omega2 * omega3
        + (m2 + m3) * l1 * l2 * s21 * (omega1 - omega2) * omega1
        + m3 * l2 * l3 * s32 * (omega2 - omega3) * omega3
    )

    # Equation 3 (theta3)
    N3 = (
        m3 * g * l3 * np.sin(theta3)
        - m3 * l2 * l3 * s23 * omega2 * omega3
        - m3 * l1 * l3 * s13 * omega1 * omega3
        + m3 * l1 * l3 * s31 * (omega1 - omega3) * omega1
        + m3 * l2 * l3 * s32 * (omega2 - omega3) * omega2
    )

    N = np.array([N1, N2, N3])

    # Solve for angular accelerations alpha
    b = -N
    alpha = np.linalg.solve(A, b)
    alpha1, alpha2, alpha3 = alpha

    # Assemble time derivative of state
    dydt = np.array([omega1, omega2, omega3, alpha1, alpha2, alpha3], dtype=float)
    return dydt


def rk4_step(f, t, y, dt, p):
    """One RK4 step."""
    k1 = f(t, y, p)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1, p)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2, p)
    k4 = f(t + dt, y + dt * k3, p)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_triple_pendulum(p: TriplePendulumParams):
    """Integrate the triple pendulum ODE using RK4."""
    n_steps = int(p.t_final / p.dt) + 1
    t = np.linspace(0.0, p.t_final, n_steps)

    y = np.zeros((n_steps, 6))
    y[0, :] = np.array([
        p.theta1_0, p.theta2_0, p.theta3_0,
        p.omega1_0, p.omega2_0, p.omega3_0
    ])

    for i in range(n_steps - 1):
        y[i + 1, :] = rk4_step(triple_pendulum_rhs, t[i], y[i, :], p.dt, p)

    return t, y


def compute_cartesian_positions(y, p: TriplePendulumParams):
    """Compute x,y positions of the three masses over time."""
    theta1 = y[:, 0]
    theta2 = y[:, 1]
    theta3 = y[:, 2]
    l1, l2, l3 = p.l1, p.l2, p.l3

    # Positions following the usual convention for chained pendula
    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)

    x2 = x1 + l2 * np.sin(theta2)
    y2 = y1 - l2 * np.cos(theta2)

    x3 = x2 + l3 * np.sin(theta3)
    y3 = y2 - l3 * np.cos(theta3)

    return x1, y1, x2, y2, x3, y3


def finite_difference(time, data):
    """Compute velocity and acceleration from position via finite differences."""
    dt = time[1] - time[0]
    vel = np.gradient(data, dt)
    acc = np.gradient(vel, dt)
    return vel, acc


def plot_trajectories_2d(time, x1, y1, x2, y2, x3, y3, savepath=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x1, y1, label="Mass 1", alpha=0.7)
    ax.plot(x2, y2, label="Mass 2", alpha=0.7)
    ax.plot(x3, y3, label="Mass 3 (tip)", linewidth=2.0)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Triple Pendulum Trajectories (2D)")
    ax.legend()
    ax.set_aspect("equal", "box")
    ax.grid(True)

    if savepath is not None:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")


def plot_trajectories_3d(time, x1, y1, x2, y2, x3, y3, savepath=None):
    # Embed y as vertical (z), to mimic x-z plane visualization
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    z1 = y1
    z2 = y2
    z3 = y3
    y_zero = np.zeros_like(x1)  # out-of-plane axis

    ax.plot(x1, y_zero, z1, label="Mass 1", alpha=0.7)
    ax.plot(x2, y_zero, z2, label="Mass 2", alpha=0.7)
    ax.plot(x3, y_zero, z3, label="Mass 3 (tip)", linewidth=2.0)

    ax.set_xlabel("x")
    ax.set_ylabel("y (out of plane)")
    ax.set_zlabel("z (vertical)")
    ax.set_title("Triple Pendulum Trajectories (3D embedding)")
    ax.legend()

    if savepath is not None:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")


def plot_kinematics_tip3_xyz(time, x3, y3, save_dir=None):
    """
    Make 9 plots: x,y,z position, velocity, acceleration of tip (mass 3).

    We embed y as z (vertical) and set out-of-plane y=0.
    """
    z3 = y3
    y_zero = np.zeros_like(x3)

    # Positions
    vx, ax = finite_difference(time, x3)
    vy, ay = finite_difference(time, y_zero)
    vz, az = finite_difference(time, z3)

    fig, axes = plt.subplots(3, 3, figsize=(12, 8), sharex=True)
    axes = axes.ravel()

    # Position
    axes[0].plot(time, x3)
    axes[0].set_ylabel("x [m]")
    axes[0].set_title("Position")

    axes[1].plot(time, y_zero)
    axes[1].set_ylabel("y [m]")

    axes[2].plot(time, z3)
    axes[2].set_ylabel("z [m]")

    # Velocity
    axes[3].plot(time, vx)
    axes[3].set_ylabel("vx [m/s]")
    axes[3].set_title("Velocity")

    axes[4].plot(time, vy)
    axes[4].set_ylabel("vy [m/s]")

    axes[5].plot(time, vz)
    axes[5].set_ylabel("vz [m/s]")

    # Acceleration
    axes[6].plot(time, ax)
    axes[6].set_ylabel("ax [m/s²]")
    axes[6].set_xlabel("t [s]")
    axes[6].set_title("Acceleration")

    axes[7].plot(time, ay)
    axes[7].set_ylabel("ay [m/s²]")
    axes[7].set_xlabel("t [s]")

    axes[8].plot(time, az)
    axes[8].set_ylabel("az [m/s²]")
    axes[8].set_xlabel("t [s]")

    for ax_ in axes:
        ax_.grid(True)

    fig.tight_layout()

    if save_dir is not None:
        fig.savefig(f"{save_dir}/tip3_kinematics_xyz.png", dpi=200, bbox_inches="tight")


def animate_triple_pendulum_2d(time, x1, y1, x2, y2, x3, y3, savepath=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal", "box")
    ax.set_xlim(min(x3) - 1.5, max(x3) + 1.5)
    ax.set_ylim(min(y3) - 1.5, max(y3) + 1.5)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Triple Pendulum (2D animation)")
    ax.grid(True)

    line_links, = ax.plot([], [], "-o", lw=2)
    trail3, = ax.plot([], [], "-", lw=1.5, alpha=0.7, color="C2")

    def init():
        line_links.set_data([], [])
        trail3.set_data([], [])
        return line_links, trail3

    def update(frame):
        x = [0, x1[frame], x2[frame], x3[frame]]
        y = [0, y1[frame], y2[frame], y3[frame]]
        line_links.set_data(x, y)
        trail3.set_data(x3[:frame + 1], y3[:frame + 1])
        return line_links, trail3

    ani = FuncAnimation(fig, update, frames=len(time), init_func=init, blit=True, interval=20)

    if savepath is not None:
        ani.save(savepath, fps=50)

    return ani


def main():
    p = TriplePendulumParams(
        t_final=10.0,
        dt=0.002,
        theta1_0=np.pi / 2,
        theta2_0=0.2,
        theta3_0=-0.3,
        omega1_0=0.0,
        omega2_0=0.0,
        omega3_0=0.0,
    )

    # Simulate
    t, y = simulate_triple_pendulum(p)
    x1, y1, x2, y2, x3, y3 = compute_cartesian_positions(y, p)

    # Plots (you can adapt paths to your results directory)
    plot_trajectories_2d(t, x1, y1, x2, y2, x3, y3, savepath="triple_traj_2d.png")
    plot_trajectories_3d(t, x1, y1, x2, y2, x3, y3, savepath="triple_traj_3d.png")
    plot_kinematics_tip3_xyz(t, x3, y3, save_dir=".")

    # Animation (2D)
    animate_triple_pendulum_2d(t, x1, y1, x2, y2, x3, y3, savepath="triple_pendulum_2d.mp4")

    plt.show()


if __name__ == "__main__":
    main()

