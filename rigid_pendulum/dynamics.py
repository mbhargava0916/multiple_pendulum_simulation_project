"""
Dynamics for a planar double pendulum with generalized coordinates q, qdot.

We write the equations in the form:

    M(q) * qddot + h(q, qdot) = tau

where:
    q    = [theta1, theta2]
    tau  = generalized torques at joints 1 and 2
    M    = 2x2 mass matrix
    h    = Coriolis/centrifugal + gravity terms
"""

import numpy as np
from .params import SystemParams


def mass_matrix(q: np.ndarray, p: SystemParams) -> np.ndarray:
    """
    Generalized mass matrix M(q) for the double pendulum.

    Assumes:
    - Two uniform rods with COM at L/2.
    """
    theta1, theta2 = q
    m1, m2 = p.body1.mass, p.body2.mass
    L1, L2 = p.body1.length, p.body2.length
    I1, I2 = p.body1.inertia_zz, p.body2.inertia_zz

    lc1 = L1 / 2.0
    lc2 = L2 / 2.0

    # Standard double-pendulum inertia terms
    M11 = (I1 + I2 +
           m1 * lc1**2 +
           m2 * (L1**2 + lc2**2 + 2 * L1 * lc2 * np.cos(theta2)))

    M12 = I2 + m2 * (lc2**2 + L1 * lc2 * np.cos(theta2))

    M22 = I2 + m2 * lc2**2

    M = np.array([[M11, M12],
                  [M12, M22]])
    return M


def nonlinear_forces(q: np.ndarray, qdot: np.ndarray, p: SystemParams) -> np.ndarray:
    """
    Combined Coriolis/centrifugal and gravity terms h(q, qdot).

    The equations here come from standard double-pendulum derivations.
    """
    theta1, theta2 = q
    dtheta1, dtheta2 = qdot
    m1, m2 = p.body1.mass, p.body2.mass
    L1, L2 = p.body1.length, p.body2.length
    lc1, lc2 = L1 / 2.0, L2 / 2.0
    g = p.sim.g

    # Gravity contributions (G1, G2)
    G1 = ((m1 * lc1 + m2 * L1) * g * np.sin(theta1) +
          m2 * lc2 * g * np.sin(theta1 + theta2))
    G2 = m2 * lc2 * g * np.sin(theta1 + theta2)

    # Coupling (Coriolis/centrifugal) contributions
    # These follow one common parametrization of the double-pendulum equations
    C1 = (-m2 * L1 * lc2 *
          (2 * dtheta1 * dtheta2 + dtheta2**2) *
          np.sin(theta2))
    C2 = (m2 * L1 * lc2 *
          dtheta1**2 *
          np.sin(theta2))

    h1 = C1 + G1
    h2 = C2 + G2

    return np.array([h1, h2])


def input_torques(t: float, q: np.ndarray, qdot: np.ndarray, p: SystemParams) -> np.ndarray:
    """
    Generalized input torques (tau).

    For the base implementation we set zero torques (free-motion under gravity),
    but this hook allows you to add:
      - Prescribed torque at joint 1
      - Control laws, etc.
    """
    _ = (t, q, qdot, p)  # unused in current version
    tau1 = 0.0
    tau2 = 0.0
    return np.array([tau1, tau2])
