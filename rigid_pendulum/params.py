from dataclasses import dataclass


@dataclass
class BodyParams:
    length: float
    mass: float
    inertia_zz: float  # planar inertia about out-of-plane axis


@dataclass
class SimParams:
    g: float
    t_end: float
    dt: float
    # You can add stabilization params later if needed
    alpha_baumgarte: float = 0.0
    beta_baumgarte: float = 0.0


@dataclass
class SystemParams:
    body1: BodyParams
    body2: BodyParams
    sim: SimParams


def default_system_params() -> SystemParams:
    """
    Returns a reasonable default double-pendulum system:
    - Two uniform rods of length 1 m and mass 1 kg
    - Gravity 9.81 m/s^2
    - Simulation 10 seconds with dt = 1e-3
    """
    L1 = 1.0
    L2 = 1.0
    m1 = 1.0
    m2 = 1.0
    I1 = m1 * L1**2 / 12.0
    I2 = m2 * L2**2 / 12.0

    body1 = BodyParams(length=L1, mass=m1, inertia_zz=I1)
    body2 = BodyParams(length=L2, mass=m2, inertia_zz=I2)
    sim = SimParams(g=9.81, t_end=10.0, dt=1e-3)

    return SystemParams(body1=body1, body2=body2, sim=sim)
