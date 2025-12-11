import numpy as np
from rigid_pendulum.params import default_system_params
from rigid_pendulum.integrator import simulate


def test_simulate_runs_short():
    p = default_system_params()

    # Shorten the simulation for the test
    p.sim.t_end = 0.1

    q0 = np.array([0.1, 0.2])
    qdot0 = np.array([0.0, 0.0])

    results = simulate(p, q0, qdot0)

    t = results["t"]
    q = results["q"]
    qdot = results["qdot"]

    assert t.ndim == 1
    assert q.shape[0] == t.shape[0]
    assert q.shape[1] == 2
    assert qdot.shape == q.shape
