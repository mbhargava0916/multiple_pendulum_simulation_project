import numpy as np
from rigid_pendulum.params import default_system_params
from rigid_pendulum.kinematics import joint_positions


def test_joint_positions_shapes():
    p = default_system_params()
    q = np.array([0.0, 0.0])
    pos = joint_positions(q, p)
    assert pos["base"].shape == (2,)
    assert pos["joint12"].shape == (2,)
    assert pos["tip2"].shape == (2,)
