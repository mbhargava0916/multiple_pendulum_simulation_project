import numpy as np
from rigid_pendulum.params import default_system_params
from rigid_pendulum.integrator import simulate
from rigid_pendulum.visualization import plot_tip_trajectory, plot_tip_y_vs_time


def test_plot_functions_smoke():
    p = default_system_params()
    p.sim.t_end = 0.05  # short sim for test
    q0 = np.array([0.1, 0.2])
    qdot0 = np.array([0.0, 0.0])

    results = simulate(p, q0, qdot0)

    # Just check that plotting runs without error
    fig1, ax1 = plot_tip_trajectory(results, p)
    fig2, ax2 = plot_tip_y_vs_time(results, p)

    assert fig1 is not None
    assert fig2 is not None
    # No need to show or save in tests
