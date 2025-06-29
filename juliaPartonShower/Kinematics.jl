include("Constants.jl")

using Roots
using LinearAlgebra

# Get the rotation matrix for two given vectors
function getRotationMatrix(v1::Vector{Float64}, v2::Vector{Float64})
    # The the cross product of two vectors
    k = cross(v1, v2)

    if norm(k) == 0
        return [1 0 0; 0 1 0; 0 0 1]
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
        push!(rotatedParticles, p)

    end
    return rotatedParticles
end

# Function to get the boost factor for the outgoing jet (new) and parent jet (old)
function boostFactor(k::Float64, new::Vector{Float64}, old::Vector{Float64})
    
    qs = new[1]^2 + new[2]^2 + new[3]^2
    q = sqrt(qs)
    Q2 = new[4]^2 - new[1]^2 - new[2]^2 - new[3]^2
    kp = k * sqrt(old[1]^2 + old[2]^2 + old[3]^2)
    kps = kp^2
    betamag = (q * new[4] - kp * sqrt(kps + Q2)) / (kps + qs + Q2)
    beta = betamag * (k / kp) * [old[1], old[2], old[3]]

    if betamag >= 0
        return beta
    else
        return [0, 0, 0]
    end

end


function boost(p::Particle, beta::Vector{Float64})

    bmag = sqrt(beta[1]^2 + beta[2]^2 + beta[3]^2)
    gamma = 1/ sqrt(1 - bmag^2)

    p.px = - gamma * beta[0] * p.E + (    1 + (gamma -1) * beta[0]^2 / bmag^2) * p.px + ( (gamma - 1) * beta[0] * beta[1] / bmag^2) * p.py +  ((gamma - 1) * beta[0] * beta[2] / bmag^2) * p.pz
    p.py = - gamma * beta[1] * p.E + ( (gamma -1) * beta[0] * beta[1] / bmag^2) * p.px + (    1 + (gamma - 1) * beta[1]^2 / bmag^2) * p.py + ( (gamma - 1) * beta[1] * beta[2] / bmag^2) * p.pz
    p.pz = - gamma * beta[2] * p.E + ( (gamma -1) * beta[0] * beta[2] / bmag^2) * p.px + ( (gamma - 1) * beta[2] * beta[1] / bmag^2) * p.py + (     1 + (gamma - 1) * beta[2]^2/ bmag^2) * p.pz
    p.E =    gamma * p.E - gamma * beta[0] * p.px - gamma * beta[1] * p.py - gamma * beta[2] * p.pz

    return p
end

# The function to numerically solve to find k
function kEq(k::Float64, p::Vector{Float64}, q::Vector{Float64}, s::Float64)
    sump = 0
    for i in range(1, length(p))

        sump = sump + sqrt(k^2 * p[i] + q[i])
        
    end
    sump = sump - s
    return sump
end

# Function to numerically solve for k
function solvekFactor(pj::Vector{Float64}, qj::Vector{Float64}, s::Float64)
    # Turn the function into one that can be parsed with the arguements into the root finder
    kEQ = (k -> kEq(k, pj, qj, s))
    sol = find_zero(kEQ, 1.01)
    return sol
end


# Function to perform global momemntum conservation
function globalMomCons(showeredParticles::Vector{Particle}, Jets::Vector{Jet})
    pj = Float64[] # The momenta of the parent parton
    qj = Float64[] # The momenta of the jets
    newqs = Vector{Float64}[] # Array to hold the outgoing jet's momentum
    oldps = Vector{Float64}[] # The progenitor's momentum
    rotms = Matrix{Float64}[] # The rotation matrices
    # Initialize the total energy
    sqrts = 0


    for jet in Jets
        # Append the 3-momentum of the Jet's progentior
        append!(pj, jet.Progenitor.px^2 + jet.Progenitor.py^2 + jet.Progenitor.pz^2)

        # Get the jet progenitor's 4 momentum
        oldp = [jet.Progenitor.px, jet.Progenitor.py, jet.Progenitor.pz, jet.Progenitor.E]

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
        rotMat = getRotationMatrix([newq[1], newq[2], newq[3]], [oldp[1], oldp[2], oldp[3]])
        push!(oldps, oldp)
        push!(newqs, newq)
        append!(qj, qj2) 
        push!(rotms, rotMat)
    end

    # Get the k factor
    k = solvekFactor(pj, qj, sqrts)

    rotatedShoweredParticles = []

    # Get the electron's and postirons and append them to the list of roated showered particles since these don't need to be rotated
    for p in showeredParticles
        if abs(p.id) == 11
            push!(rotatedShoweredParticles, p)
        end
    end

    # Check if any of the jets have been radiated, if not, do not perform the boost and append the progenitor of each jet
    ifRadiated = any(length(jet.AllParticles) > 1 for jet in Jets)
    if ifRadiated == false
        for j in Jets
            push!(rotatedShoweredParticles, j.AllParticles[1])
        end
    else
        for (i, jet) in enumerate(Jets)
            
            # Get the boost factor
            beta = boostFactor(k, newqs[i], oldps[i])

            # Iterate through the jet's particles and rotate and boost each one
            for p in jet.AllParticles
                rotated = rotated(p, rotms[i])
                pboosted = boost(rotated, beta)
                push!(rotatedShoweredParticles, pboosted)

            end
        end
    end
    return rotatedShoweredParticles
end


# Function to reconstruct the sudakov basis for the entire tree with the progenitor being the starting root 
function reconSudakovBasis(prog::Particle, progPart::Particle)
    if prog.status == 1
        return
    end
    current = prog.children[1]
    parent = prog
    stack = [prog]

    # If the stack is empty and the currently selected particle has no children, i.e. final state, then stop the tree search
    while length(stack) != 0 || current.status == -1

        # If the current particle is initial state, i.e. there is an emisison, then append the current particle to the stack
        # then calculate the emisison of the current particle and then select the next particle as the first child in the 
        # list of children.
        if current.status == -1
            push!(stack, current)
            calculatePhysicals(current, prog, progPart, parent)
            current = current.children[1]
        
        # If the current particle is final state, i.e. no emission, then calculate its phyiscals, then select the next particle
        # as the second child of its parent and remove this parent of the stack of particles
        elseif current.status == 1
            calculatePhysicals(current, prog, progPart, parent)
            current = pop!(stack)
            current = current.children[2]
        end

    end


end

# Function to actually calculate the phyiscals of the sudakov basis for a given particle
function calculatePhysicals(part::Particle, prog::Particle, progPart::Particle, parent::Particle)

    pdotn = dot4Vec([prog.px, prog.py, prog.pz, prog.E], [progPart.px, progPart.py, progPart.pz, progPart.pz])

    alpha = parent.alpha * part.z
    part.alpha = alpha
    qi2 = part.virtuality
    kT = [part.pT * cos(part.phi), part.pT * sin(part.phi), 0 , 0] # pT = (px, py, pz, E)
    part.qT = (parent.px * part.z - kT[1], parent.py * part.z - kT[2], 0, 0) # qT = (px, py, pz, E)
    betai = (qi2 - part.alpha^2 * part.m^2 - (-part.qT[1]^2 - part.qT[2])) / (2 * part.alpha * pdotn)

    part.px = alpha * prog.px + betai * progPart.px + part.qT[1]
    part.py = alpha * prog.py + betai * progPart.py + part.qT[2]
    part.pz = alpha * prog.pz + betai * progPart.pz
    part.E = alpha * prog.E + betai * progPart.E

end

function dot4Vec(v1::Vector{Float64}, v2::Vector{Float64})
    return v1[4] * v2[4] - v1[1] * v2[1] - v1[2] * v2[2] - v1[3] * v2[3]
end

# Function to find the color partner of a particle
function findColorPartner(parti::Particle, particles::Vector{Particle})

    partner = parti

    for pc in particles
        if patri.color == pc.antiColor && parti.antiColor == pc.color
            partner = pc
        end
    end 

end
