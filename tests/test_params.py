from rigid_pendulum.params import default_system_params


def test_default_system_params():
    p = default_system_params()
    assert p.body1.length > 0.0
    assert p.body2.length > 0.0
    assert p.sim.dt > 0.0
    assert p.sim.t_end > 0.0
