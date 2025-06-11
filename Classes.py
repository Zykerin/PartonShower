from dataclasses import dataclass

# A class for the emission info
@dataclass
class emissioninfo:
    t: float
    z: float
    Ptsq:float
    Vmsq: float
    phi: float
    Generated: bool
    ContinueEvolve: bool

# A class for particles
@dataclass
class Particle:
    typ: int
    status: int
    t_at_em: float
    z_at_em: float
    Pt: float
    Px: float
    Py: float
    Pz: float
    m: float
    E: float
    phi: float
    ContinueEvolution: bool = True


# A dataclass for jets
@dataclass
class Jet:
    Progenitor: Particle
    Particles: list[Particle]


# A data class for an event
@dataclass
class Event:
    AllParticles: list[Particle]
    Jets: list[Jet] 