"""
Visualization utilities: plots and Matplotlib animation.

These directly support the "Results" section of the report.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)

from .params import SystemParams
from .kinematics import joint_positions, tip_position_3d


# ---------- Basic 2D trajectory plots ----------

def plot_tip_trajectory(results: dict, p: SystemParams, savepath=None):
    """
    Plot the x-y trajectory of the tip of body 1 (joint12, orange)
    and tip of body 2 (tip2, blue) on the same figure.
    """
    t = results["t"]
    q_hist = results["q"]

    joint_positions_list = []
    tip_positions_list = []
    for i in range(len(t)):
        pos = joint_positions(q_hist[i], p)
        joint_positions_list.append(pos["joint12"])
        tip_positions_list.append(pos["tip2"])

    joint_positions_arr = np.array(joint_positions_list)  # tip 1
    tip_positions_arr = np.array(tip_positions_list)      # tip 2

    fig, ax = plt.subplots()
    # Tip 2 in blue (default)
    ax.plot(tip_positions_arr[:, 0], tip_positions_arr[:, 1], label="Tip 2")
    # Tip 1 in orange
    ax.plot(joint_positions_arr[:, 0], joint_positions_arr[:, 1],
            color="tab:orange", label="Tip 1")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Tip trajectories of body 1 and body 2 (x–y plane)")
    ax.grid(True)
    ax.legend()

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_tip_y_vs_time(results: dict, p: SystemParams, savepath=None):
    """
    Plot vertical position of the tip of body 2 vs time.
    """
    t = results["t"]
    q_hist = results["q"]

    tip_positions = np.array([
        joint_positions(q_hist[i], p)["tip2"] for i in range(len(t))
    ])

    y = tip_positions[:, 1]

    fig, ax = plt.subplots()
    ax.plot(t, y)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("y_tip (m)")
    ax.set_title("Vertical position of tip of body 2")
    ax.grid(True)

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    return fig, ax


# ---------- 3D kinematics (x,y,z, vx,vy,vz, ax,ay,az) ----------

def compute_tip_kinematics_3d(results: dict, p: SystemParams):
    """
    Compute 3D position, velocity, and acceleration of the tip of body 2.

    Position is obtained via kinematics (tip_position_3d).
    Velocity and acceleration are computed by finite differences.
    """
    t = results["t"]
    q_hist = results["q"]
    dt = t[1] - t[0]

    n = len(t)

    pos = np.zeros((n, 3))
    for i in range(n):
        pos[i, :] = tip_position_3d(q_hist[i], p)

    vel = np.zeros_like(pos)
    acc = np.zeros_like(pos)

    # Central differences for interior points
    vel[1:-1] = (pos[2:] - pos[:-2]) / (2.0 * dt)
    vel[0] = (pos[1] - pos[0]) / dt
    vel[-1] = (pos[-1] - pos[-2]) / dt

    acc[1:-1] = (vel[2:] - vel[:-2]) / (2.0 * dt)
    acc[0] = (vel[1] - vel[0]) / dt
    acc[-1] = (vel[-1] - vel[-2]) / dt

    return t, pos, vel, acc


def plot_kinematics_xyz(results: dict, p: SystemParams, save_dir=None):
    """
    Produce 9 plots total:

      1) x(t), y(t), z(t)    - position components
      2) vx(t), vy(t), vz(t) - velocity components
      3) ax(t), ay(t), az(t) - acceleration components

    Each set is plotted as 3 subplots in one figure.

    If save_dir is provided (Path or str), saves:
      - position_xyz.png
      - velocity_xyz.png
      - acceleration_xyz.png
    """
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)

    t, pos, vel, acc = compute_tip_kinematics_3d(results, p)

    # --- Position ---
    fig_pos, axes_pos = plt.subplots(3, 1, sharex=True, figsize=(6, 8))
    labels = ["x", "y", "z"]
    for i in range(3):
        axes_pos[i].plot(t, pos[:, i])
        axes_pos[i].set_ylabel(f"{labels[i]} (m)")
        axes_pos[i].grid(True)
    axes_pos[-1].set_xlabel("time (s)")
    fig_pos.suptitle("Tip position components vs time")

    if save_dir is not None:
        fig_pos.savefig(save_dir / "position_xyz.png",
                        dpi=300, bbox_inches="tight")

    # --- Velocity ---
    fig_vel, axes_vel = plt.subplots(3, 1, sharex=True, figsize=(6, 8))
    labels_v = ["vx", "vy", "vz"]
    for i in range(3):
        axes_vel[i].plot(t, vel[:, i])
        axes_vel[i].set_ylabel(f"{labels_v[i]} (m/s)")
        axes_vel[i].grid(True)
    axes_vel[-1].set_xlabel("time (s)")
    fig_vel.suptitle("Tip velocity components vs time")

    if save_dir is not None:
        fig_vel.savefig(save_dir / "velocity_xyz.png",
                        dpi=300, bbox_inches="tight")

    # --- Acceleration ---
    fig_acc, axes_acc = plt.subplots(3, 1, sharex=True, figsize=(6, 8))
    labels_a = ["ax", "ay", "az"]
    for i in range(3):
        axes_acc[i].plot(t, acc[:, i])
        axes_acc[i].set_ylabel(f"{labels_a[i]} (m/s²)")
        axes_acc[i].grid(True)
    axes_acc[-1].set_xlabel("time (s)")
    fig_acc.suptitle("Tip acceleration components vs time")

    if save_dir is not None:
        fig_acc.savefig(save_dir / "acceleration_xyz.png",
                        dpi=300, bbox_inches="tight")

    return (fig_pos, fig_vel, fig_acc)


# ---------- 2D animation (still available if you want it) ----------

def animate_pendulum(results: dict, p: SystemParams, savepath=None):
    """
    2D animation of the double pendulum in the x–y plane.

    Shows:
      - blue solid lines = current links (pendulum)
      - orange dashed line = trail of Tip 1
      - green dashed line = trail of Tip 2
    """
    t = results["t"]
    q_hist = results["q"]
    dt = t[1] - t[0]
    n_steps = len(t)
    t_end = t[-1]

    # Precompute 2D positions
    base_xy = np.zeros((n_steps, 2))
    joint_xy = np.zeros((n_steps, 2))  # tip 1
    tip2_xy = np.zeros((n_steps, 2))   # tip 2

    for i in range(n_steps):
        pos = joint_positions(q_hist[i], p)
        base_xy[i] = pos["base"]
        joint_xy[i] = pos["joint12"]
        tip2_xy[i] = pos["tip2"]

    # Downsample frames for real-time-ish animation
    target_frames = int(30 * t_end)
    if target_frames <= 0:
        target_frames = n_steps
    frame_step = max(1, n_steps // target_frames)
    frame_indices = np.arange(0, n_steps, frame_step)

    fig, ax = plt.subplots()
    ax.set_aspect("equal", "box")

    L_total = p.body1.length + p.body2.length
    margin = 0.2 * L_total
    ax.set_xlim(-L_total - margin, L_total + margin)
    ax.set_ylim(-L_total - margin, L_total + margin)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Double pendulum motion (2D)")

    # Current links
    link1_line, = ax.plot([], [], lw=2, color="tab:blue")
    link2_line, = ax.plot([], [], lw=2, color="tab:blue")

    # Trails
    tip1_trail, = ax.plot([], [], linestyle="--",
                          linewidth=1.5, color="tab:orange", label="Tip 1 trail")
    tip2_trail, = ax.plot([], [], linestyle="--",
                          linewidth=1.5, color="tab:green", label="Tip 2 trail")

    ax.legend()

    def update(frame_idx):
        i = frame_indices[frame_idx]

        b = base_xy[i]
        j = joint_xy[i]
        t2 = tip2_xy[i]

        # current links
        link1_line.set_data([b[0], j[0]],
                            [b[1], j[1]])
        link2_line.set_data([j[0], t2[0]],
                            [j[1], t2[1]])

        # trails up to now
        j_trail = joint_xy[: i + 1]
        t2_trail = tip2_xy[: i + 1]
        tip1_trail.set_data(j_trail[:, 0], j_trail[:, 1])
        tip2_trail.set_data(t2_trail[:, 0], t2_trail[:, 1])

        return link1_line, link2_line, tip1_trail, tip2_trail

    def init():
        return update(0)

    ani = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        init_func=init,
        blit=False,                  # safer
        interval=1000 * dt * frame_step,
    )

    if savepath is not None:
        ani.save(savepath, fps=30)

    return fig, ani

# ---------- NEW: 3D animation with trails ----------

def animate_pendulum_3d(results: dict, p: SystemParams, savepath=None):
    """
    3D Matplotlib animation of the double pendulum.

    Planar motion embedded in x–z plane so it visually swings up/down:
        x_3D = x_physical
        y_3D = 0
        z_3D = y_physical

    Trails are drawn for Tip 1 and Tip 2.
    No blitting (3D + blit often produces blank plots).
    """
    t = results["t"]
    q_hist = results["q"]
    dt = t[1] - t[0]
    n_steps = len(t)
    t_end = t[-1]

    # Precompute 3D positions: (x, 0, y) so z shows vertical motion
    base_xyz = np.zeros((n_steps, 3))
    joint_xyz = np.zeros((n_steps, 3))
    tip2_xyz = np.zeros((n_steps, 3))

    for i in range(n_steps):
        pos = joint_positions(q_hist[i], p)
        base2d = pos["base"]
        joint2d = pos["joint12"]
        tip2_2d = pos["tip2"]

        base_xyz[i] = [base2d[0], 0.0, base2d[1]]
        joint_xyz[i] = [joint2d[0], 0.0, joint2d[1]]
        tip2_xyz[i] = [tip2_2d[0], 0.0, tip2_2d[1]]

    # Downsample frames for real-time-ish animation
    target_frames = int(30 * t_end)
    if target_frames <= 0:
        target_frames = n_steps
    frame_step = max(1, n_steps // target_frames)
    frame_indices = np.arange(0, n_steps, frame_step)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    L_total = p.body1.length + p.body2.length
    margin = 0.2 * L_total
    ax.set_xlim(-L_total - margin, L_total + margin)
    ax.set_ylim(-L_total - margin, L_total + margin)  # depth
    ax.set_zlim(-L_total - margin, L_total + margin)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("depth (m)")
    ax.set_zlabel("z (m)")  # vertical on screen
    ax.set_title("Double pendulum motion (3D view with vertical swing)")

    # View so x is horizontal and z vertical on screen
    ax.view_init(elev=20, azim=-90)

    # Current links
    link1_line, = ax.plot([], [], [], lw=2, color="tab:blue")
    link2_line, = ax.plot([], [], [], lw=2, color="tab:blue")

    # Trails for tip1 and tip2
    tip1_trail, = ax.plot([], [], [], linestyle="--",
                          linewidth=1.5, color="tab:orange", label="Tip 1 trail")
    tip2_trail, = ax.plot([], [], [], linestyle="--",
                          linewidth=1.5, color="tab:green", label="Tip 2 trail")

    ax.legend()

    def update(frame_idx):
        i = frame_indices[frame_idx]

        b = base_xyz[i]
        j = joint_xyz[i]
        t2 = tip2_xyz[i]

        # Current links
        link1_line.set_data_3d([b[0], j[0]],
                               [b[1], j[1]],
                               [b[2], j[2]])
        link2_line.set_data_3d([j[0], t2[0]],
                               [j[1], t2[1]],
                               [j[2], t2[2]])

        # Trails up to current frame
        j_trail = joint_xyz[: i + 1]
        t2_trail = tip2_xyz[: i + 1]

        tip1_trail.set_data_3d(j_trail[:, 0],
                               j_trail[:, 1],
                               j_trail[:, 2])
        tip2_trail.set_data_3d(t2_trail[:, 0],
                               t2_trail[:, 1],
                               t2_trail[:, 2])

        return link1_line, link2_line, tip1_trail, tip2_trail

    def init():
        # draw first frame
        return update(0)

    ani = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        init_func=init,
        blit=False,                    # <- IMPORTANT
        interval=1000 * dt * frame_step,
    )

    if savepath is not None:
        ani.save(savepath, fps=30)

    return fig, ani


def plot_tip_trajectories_3d(results: dict, p: SystemParams, savepath=None):
    """
    Static 3D plot showing ONLY the trajectories (trails) of Tip 1 and Tip 2.

    No pendulum links are drawn; just the paths, similar to the 3D animation
    but as a single figure for the report.
    """
    t = results["t"]
    q_hist = results["q"]
    n_steps = len(t)

    # Precompute 3D positions in x–z plane: (x, 0, y)
    joint_xyz = np.zeros((n_steps, 3))
    tip2_xyz = np.zeros((n_steps, 3))

    for i in range(n_steps):
        pos = joint_positions(q_hist[i], p)
        joint2d = pos["joint12"]
        tip2_2d = pos["tip2"]

        joint_xyz[i] = [joint2d[0], 0.0, joint2d[1]]
        tip2_xyz[i] = [tip2_2d[0], 0.0, tip2_2d[1]]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    L_total = p.body1.length + p.body2.length
    margin = 0.2 * L_total
    ax.set_xlim(-L_total - margin, L_total + margin)
    ax.set_ylim(-L_total - margin, L_total + margin)
    ax.set_zlim(-L_total - margin, L_total + margin)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("depth (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("Tip trajectories of body 1 and body 2 (3D)")

    # Same viewing angle as animation
    ax.view_init(elev=20, azim=-90)

    ax.plot(joint_xyz[:, 0], joint_xyz[:, 1], joint_xyz[:, 2],
            linestyle="--", linewidth=1.5, color="tab:orange", label="Tip 1 trail")
    ax.plot(tip2_xyz[:, 0], tip2_xyz[:, 1], tip2_xyz[:, 2],
            linestyle="--", linewidth=1.5, color="tab:green", label="Tip 2 trail")

    ax.legend()

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    return fig, ax
