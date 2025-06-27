
# The structure for an emission
struct Emission
   t::Float64
   z::Float64
   pTsq::Float64
   phi::Float64
   Generated::Bool
   ContinueEvolution::Bool

end

# The structure for a particle.
struct Particle
    id::Int
    status::Int
    t::Float64
    z::Float64
    m::Float64
    px::Float64
    py::Float64
    pz::Float64
    E::Float64
    phi::Float64
    continueEvolution::Bool
    parent::Vector{Particle}
    children::Vector{Particle}


end

struct Jets
   AllParticles::Vector{Particle}
   Progenitor::Particle
end


struct Event
   AllParticles::Vector{Particle}
   Jets::Vector


end


