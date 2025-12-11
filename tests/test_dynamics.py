import numpy as np
from rigid_pendulum.params import default_system_params
from rigid_pendulum.dynamics import mass_matrix, nonlinear_forces, input_torques


def test_mass_matrix_positive_definite():
    p = default_system_params()
    q = np.array([0.1, -0.2])
    M = mass_matrix(q, p)
    # 2x2 symmetric, should have positive diag and positive eigenvalues
    assert M.shape == (2, 2)
    eigs = np.linalg.eigvals(M)
    assert (eigs > 0).all()


def test_nonlinear_forces_shape():
    p = default_system_params()
    q = np.array([0.1, -0.2])
    qdot = np.array([0.3, -0.1])
    h = nonlinear_forces(q, qdot, p)
    assert h.shape == (2,)


def test_input_torques_zero():
    p = default_system_params()
    q = np.array([0.0, 0.0])
    qdot = np.array([0.0, 0.0])
    tau = input_torques(0.0, q, qdot, p)
    assert tau.shape == (2,)
    assert (tau == 0.0).all()
