"""
Time integration for the double pendulum.

We integrate the second-order system:

    M(q) qddot + h(q, qdot) = tau

using a simple semi-implicit Euler scheme:

    qdot_{n+1} = qdot_n + qddot_n * dt
    q_{n+1}    = q_n    + qdot_{n+1} * dt
"""

import numpy as np
from .params import SystemParams
from .dynamics import mass_matrix, nonlinear_forces, input_torques


def simulate(p: SystemParams,
             q0: np.ndarray,
             qdot0: np.ndarray) -> dict:
    """
    Run the simulation from t=0 to t_end with time step dt.

    Parameters
    ----------
    p : SystemParams
        System and simulation parameters.
    q0 : np.ndarray
        Initial generalized coordinates [theta1, theta2].
    qdot0 : np.ndarray
        Initial generalized velocities [dtheta1, dtheta2].

    Returns
    -------
    dict with keys:
      - 't': time array, shape (N,)
      - 'q': state history, shape (N, 2)
      - 'qdot': velocity history, shape (N, 2)
    """
    dt = p.sim.dt
    t_end = p.sim.t_end
    n_steps = int(t_end / dt) + 1

    q_hist = np.zeros((n_steps, 2))
    qdot_hist = np.zeros((n_steps, 2))
    t_hist = np.linspace(0.0, t_end, n_steps)

    q = q0.copy()
    qdot = qdot0.copy()

    q_hist[0] = q
    qdot_hist[0] = qdot

    for i in range(1, n_steps):
        t = t_hist[i - 1]

        # 1) Evaluate dynamics
        M = mass_matrix(q, p)
        h = nonlinear_forces(q, qdot, p)
        tau = input_torques(t, q, qdot, p)

        # 2) Solve for accelerations
        qddot = np.linalg.solve(M, tau - h)

        # 3) Semi-implicit Euler update
        qdot = qdot + qddot * dt
        q = q + qdot * dt

        q_hist[i] = q
        qdot_hist[i] = qdot

    return {"t": t_hist, "q": q_hist, "qdot": qdot_hist}
