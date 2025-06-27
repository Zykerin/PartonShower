include("Constants.jl")
using Roots

using LinearAlgebra

# Get the rotation matrix for two given vectors
function getRotationMatrix(v1::Vector{Float64}, v2::Vector{Float64})
    # The the cross product of two vectors
    k = cross(v1, v2)

    if norm(k) == 0
        return I
    else
        # Get unit vector of k
        k = k / norm(k)
        # Get the k matrix
        K = [0 -k[3] k[2]; k[3] 0 -k[1]; -k[2] k[1] 0]
        
        theta = acos(clamp(dot(v1/norm(v1), v2/norm(v2)), -1, 1))

        # Get the rotation matrix
        rotMat = I + sin(theta) * K + (1 - cos(theta)) * K^2
        return rotMat
    end
end

# Function to rotate a given particle with a given rotation matrix
function rotate(p::Particle, rotMat::Matrix)
    pvec = [p.px, p.py, p.pz]
    rotVec = rotMat * pvec
    p.px = rotVec[1]
    p.py = rotVec[2]
    p.pz = rotVec[3]
    return p
end


# Function to rotate the particle to allign with the mother particle's momentum in the lab frame
function rotateMomentaLab(p::Particle, particles::Vector{Particle})

    rotatedParticles = []

    pmag = sqrt(p.px^2 + p.py^2 + p.pz^2)
    m = getRotationMatrix([0, 0, pmag], [p.px, p.py, p.pz])
    for p in particles
        v = [p.px, p.py, p.pz]
        rotatedVec = m * v
        p.px = rotatedVec[1]
        p.py = rotatedVec[2]
        p.pz = rotatedVec[3]
        append!(rotatedParticles, p)

    end
    return rotatedParticles
end

# Function to get the boost factor for the outgoing jet (new) and parent jet (old)
function boostFactor(k::Float64, new, old)
    
    qs = new[1]^2 + new[2]^2 + new[3]^2
    q = sqrt(qs)
    Q2 = new[4]^2 - new[1]^2 - new[2]^2 - new[3]^2
    kp = k * sqrt(old[1]^2 + old[2]^2 + old[3]^2)
    kps = kp^2
    betmag = (q * new[4] - kp * sqrt(kps + Q2)) / (kps + qs + Q2)
    beta = betamag * (k / kp) * [old[1], old[2], old[3]]

    if betamag >= 0
        return beta
    else
        return [0, 0, 0]
    end

end


function boost(p::Paticle, beta::Vector{Float64})

    bmag = sqrt(beta[1]^2 + beta[2]^2 + beta[3]^2)
    gamma = 1/ sqrt(1 - bmag^2)

    p.px = - gamma * beta[0] * p.E + (    1 + (gamma -1) * beta[0]^2 / bmag^2) * p.px + ( (gamma - 1) * beta[0] * beta[1] / bmag^2) * p.py +  ((gamma - 1) * beta[0] * beta[2] / bmag^2) * p.pz
    p.py = - gamma * beta[1] * p.E + ( (gamma -1) * beta[0] * beta[1] / bmag^2) * p.px + (    1 + (gamma - 1) * beta[1]^2 / bmag^2) * p.py + ( (gamma - 1) * beta[1] * beta[2] / bmag^2) * p.pz
    p.pz = - gamma * beta[2] * p.E + ( (gamma -1) * beta[0] * beta[2] / bmag^2) * p.px + ( (gamma - 1) * beta[2] * beta[1] / bmag^2) * p.py + (     1 + (gamma - 1) * beta[2]^2/ bmag^2) * p.pz
    p.E =    gamma * p.E - gamma * beta[0] * p.px - gamma * beta[1] * p.py - gamma * beta[2] * p.pz


end

# The function to numerically solve to find k
function kEq(k, p, q, s)
    sump = 0
    for i in range(1, len(p))

        sump = sump + sqrt(k^2 * p[i] + q[i])
        
    end
    sump = sump - s
    return sump
end

# Function to numerically solve for k
function solvekFactor(pj, qj, s)
    # Turn the function into one that can be parsed with the arguements into the root finder
    kEQ = (k -> kEq(k, pj, qj, s))
    sol = find_zero(kEQ, 1.01)
    return sol
end


# Function to perform global momemntum conservation
function globalMomCons(showeredParticles, Jets)
    pj = [] # The momenta of the parent parton
    qj = [] # The momenta of the jets
    newqs = [] # Array to hold the outgoing jet's momentum
    oldps = [] # The progenitor's momentum
    rotms = [] # The rotation matrices
    # Initialize the total energy
    sqrts = 0


    for jet in Jets
        # Append the 3-momentum of the Jet's progentior
        append!(pj, jet.Progenitor.px^2 + jet.Progenitor.py^2 + jet.Progenitor.pz^2)

        # Get the jet progenitor's 4 momentum
        oldp = [jet.Progenitor.px, jet.progenitor.py, jet.progenitor.pz, jet.progenitor.E]

        # Add this jet's progenitor's energy
        sqrts += jet.Progenitor.E

        # Get the total jet's momentum after showering
        newq = [0, 0, 0, 0]
        for p in jet.AllParticles
            newq += [p.px, p.py, p.pz, p.E]
        end

        # Calculate the momentum and append it to the list of jet momentum
        qj2 = newq[4]^2 - newq[1]^2 - newq[2]^2 - newq[3]^2
        if isnan(qj2)
            qj2 = 0
        end
        rotMat = getRotationMatrix([newq[1], newq[2], newq[3]], [oldp[1], old[2], oldp[3]])
        append!(oldps, [oldp])
        append!(newqs, [newq])
        append!(qj, qj2) 
        append!(rotms, [rotMat])
    end

    # Get the k factor
    k = solvekFactor(pj, qj, sqrts)

    rotatedShoweredParticles = []

    # Get the electron's and postirons and append them to the list of roated showered particles since these don't need to be rotated
    for p in showeredParticles
        if abs(p.id) == 11
            append!(rotatedShoweredParticles, p)
        end
    end

    # Check if any of the jets have been radiated, if not, do not perform the boost and append the progenitor of each jet
    ifRadiated = any(length(jet.AllParticles) > 1 for jet in Jets)
    if ifRadiated == false
        for j in Jets
            append!(rotatedShoweredParticles, j.AllParticles[1])
        end
    else
        for (i, jet) in enumerate(Jets)
            
            # Get the boost factor
            beta = boostFactor(k, newqs[i], oldps[i])

            # Iterate through the jet's particles and rotate and boost each one
            for p in jet.AllParticles
                rotated = rotated(p, rotms[i])
                pboosted = boost(rotated, beta)
                append(rotatedShoweredParticles, pboosted)

            end
        end
    end
    return rotatedShoweredParticles
end