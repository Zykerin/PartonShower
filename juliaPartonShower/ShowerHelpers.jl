using Roots
include("Structures.jl")
include("Constants.jl")

# Function to get the transverse momentum squared
# The masses variable should be an array with [mu, Qg]
function transversemmSquared(t::Float64, z::Float64, branchType::Int, masses)
    # Option for gluon radiating, either g -> gg or g -> qqbar
   if branchType == 1 || branchType == 3
       return z^2 * (1-z)^2 * t - masses[0]^2
    # Option for quark radiating, only q -> qg
   elseif branchType == 2
       return (1-z)^2 * (z^2 * t - masses[0]^2) - z * masses[1]^2
   end
end

# Function for the overestimate of the z integral limits
function zBounds(masses::Vector{Float64}, t::Float64, branchType::Int)
   mu::Float64 = masses[0]
   Qg::Floay64 = masses[1]

   if branchType == 1 || branchType == 3
       return 1 - sqrt( (mu^2 + pT2min)/ t), sqrt((mu^2 + pT2min)/t)
   elseif branchType == 2
       return 1 - sqrt((Qg^2 + pT2min)/t), sqrt((mu^2 + pT2min)/t)
   end
end


# Function for the initial evolution scale
function EvolutionScale(p1::Particle, p2::Particle)
    Q2 = (p1.E + p2.E)^2 - (p1.px + p2.px)^2 - (p1.py + p2.py)^2 - (p1.pz + p2.pz)^2

    b = p1.m^2/Q2
    c = p2.m^2/Q2
    lam = sqrt(1+b^2 + c^2 - 2 * b - 2*c -2*b*c)
    ktildb = 0.5 * (1 + b - c + lam)
    ktildc = 0.5 * (1 - b + c + lam)
    QTilde = [sqrt(Q2 * ktildb), sqrt(Q2 * ktildc)]
    return QTilde
end

# The emission scale function.
# Need to solve this function for 0 to find the next emisison scale
function E(t::Float64, Q::Float64, R1::Float64, aSover::Float64, tGamma::Function, masses::Vector{Float64}, branchType::Int8)
    zup, zlow = zBounds(masses, t, branchType)
    r = tGamma(zup, aSover) - tGamma(zlow, aSover)
    return log(t/Q^2) - log(R1)/r
end

# The function to determine the next emission scale by numerically solving the emission scale functions E(t)
function tEmission(Q::Float64, R1::Float64, aSover::Float64, tmin::Float64, tGamma::Function, masses::Vector{Float64}, branchType::Int8)

    E2 = (t -> E(t, Q, R1, aSover, tGamma, masses, branchType))
    t::Float64 = find_zero(E2, (tmin, Q^2))

    continueEvolve = true
    if abs(E2(t)) > 1E-4
        continueEvolve = false
    end

    return t, continueEvolve

end

# Function to get the z emission value
function zEmission(t::Float64, R2::Float64, aSover::Float64, tGamma::Function, inversetGamma::Function, masses::Vector{Float64}, branchType::Int8)
    zup, zlow = zBounds(masses, t, branchType)
    z = inversetGamma( tGamma(zlow, aSover) + R2 * (tGamma(zup, aSover) - tGamma(zlow, aSover)), aSover )
    return z
end