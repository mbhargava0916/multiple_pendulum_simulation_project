"""
Kinematics for a planar double pendulum.

Generalized coordinates:
    q = [theta1, theta2]

Convention (matches the dynamics formulas):
  - theta1, theta2 are measured from the vertical downward direction.
  - When theta = 0, a link hangs straight down along the negative y-axis.
  - Positive theta rotates counterclockwise.
  - Gravity acts in negative y-direction.

Coordinates:
  For a link of length L and angle theta (from vertical):
    x =  L * sin(theta)
    y = -L * cos(theta)
"""

import numpy as np
from .params import SystemParams


def joint_positions(q: np.ndarray, p: SystemParams) -> dict:
    """
    Compute positions of the base, joint between bodies, and tip of body 2
    in 2D (x, y).

    Parameters
    ----------
    q : np.ndarray
        Shape (2,), generalized coordinates [theta1, theta2].
    p : SystemParams

    Returns
    -------
    dict with keys: 'base', 'joint12', 'tip2', each a (2,) np.array.
    """
    theta1, theta2 = q
    L1 = p.body1.length
    L2 = p.body2.length

    base = np.array([0.0, 0.0])

    # First link end (joint between body 1 and 2)
    joint12 = base + L1 * np.array([
        np.sin(theta1),
        -np.cos(theta1),
    ])

    # Second link tip
    theta12 = theta1 + theta2
    tip2 = joint12 + L2 * np.array([
        np.sin(theta12),
        -np.cos(theta12),
    ])

    return {"base": base, "joint12": joint12, "tip2": tip2}


def tip_position_3d(q: np.ndarray, p: SystemParams) -> np.ndarray:
    """
    3D position of the tip of body 2.

    Motion is planar in x–y, so:
        r_tip = [x, y, 0]^T
    """
    pos2d = joint_positions(q, p)["tip2"]
    return np.array([pos2d[0], pos2d[1], 0.0])
