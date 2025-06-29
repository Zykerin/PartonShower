include("ShowerHelpers.jl")
include("SplittingFunctions.jl")
include("Kinematics.jl")
include("Constants.jl")

using Random 


Random.seed!(12345) # Set the seed for easier testing

function generateEmissions(Q::Float64, Qcut::Float64, aSover::Float64, branchType::Int8, masses::Vector{Float64})

    R1 = rand()
    R2 = rand()
    R3 = rand()
    R4 = rand()
    
    ems = Emission(0, 1, 0, 0, true, true)

    # Check for the branch type and get the appropiate z and t emissions
    if branchType == 1
        tmin = 3.99 * (mu[0]^2 + Qcut)
        ems.t, ems.ContinueEvolution = tEmission(Q, R1, aSover, tmin, Intgg, masses, branchType)
        ems.z = zEmission(ems.t, R2, aSover, Intgg, InvInt_gg, masses, branchType)
    elseif branchType == 2
        t0 = mu[0]^2 + Qcut
        t1 = mu[1]^2 + Qcut
        tmin = 0.99 * (2 * sqrt(t1 * t0) + t1 + t0)
        ems.t, ems.ContinueEvolution = tEmission(Q, R1, aSover, tmin, Intqq, masses, branchType)
        ems.z = zEmission(ems.t, R2, aSover, Intqq, InvInt_qq, masses, branchType)
    elseif branchType == 3
        tmin = 3.99 * (mu[0]^2 + Qcut)
        ems.t, ems.ContinueEvolution = tEmission(Q, R1, aSover, tmin, Intgq, masses, branchType)
        ems.z = zEmission(ems.t, R2, aSover, Intgq, InvInt_gq, masses, branchType)
    end
    
    # If the evoltion has been terminated, return the emission struc
    if ems.ContinueEvolution == false
        ems.z = 1
        return ems
    end

    # Get the transverse momentum squared
    ems.pTsq = transversemmSquared(ems.t, ems.z, branchType, masses)

    # Apply the minimum transverse momentum cut
    if ems.pTsq < Qcut 
        ems.Generated = false
    end

    # Apply the overestimate of the splitting function cut according to the branch
    if branchType == 1
       if R3 > Pgg(ems.z) > Pgg_over(ems.z)
            ems.Generated = false
       end
    elseif branchType == 2
        if R3 > Pqq(ems.z) > Pqq_over(ems.z)
            ems.Generated = false
       end
    elseif branchType == 3
        if R3 > Pgq(ems.z) > Pgq_over(ems.z)
            ems.Generated = false
       end
    end

    # Apply the alphaS oversitmate cut
    if R4 > getAlphaS(ems.t, ems.z, Qcut) / aSover
        ems.Generated = false
    end

    ems.phi = (2 * rand() - 1) * pi

    if ems.Generated == false
        ems.z = 1
    end

    return ems

end



# Functionn to evolve a specific particle given its children
function evolveParticle(pa::Particle, pb::Particle, pc::Particle, Qcut::Float64, aSover::Float64)

    tcuttoff = 4 # The curtoff for the evolution variable

    #emission = Emission(pa.t, 1, 0, 0, true, true) # Base values for the emisisons

    Q::Float64 = sqrt(pa.t) * pa.z # Get the initial scale for this evolution
    masses = [0, 0] # Get the masses which are set to 0 for now

    # This loops until either an emission is accepted and generated or until the evolution is terminated
    while true
        # Condition if the evolution scale is below the cuttoff to break
        if Q < sqrt(tcuttoff * Qcut^2)
            pa.ContinueEvolution = false
            return
        end
        # Branch for quark emiting. Currently only q -> qg
        if abs(pa.id) < 6 && abs(pa.id) > 0
            branchType = 2
            pb.m = mufunc(pa.m, Qg(pa.m))
            pc.m = Qg(pa.m)
            emission = generateEmissions(Q, Qcut, aSover, branchType, masses)
            pb.id = pa.id
            pc.id = 21
        # Condition for gluon emittion which needs competition
        elseif abs(pa.id) == 21
            branchType = 1
            pb.m = Qg(pa.m)
            pc.m = Qg(pa.m)

            emission = generateEmissions(Q, Qcut, aSover, branchType, masses)

            pb.id = 21
            pc.id = 21

            if emission.continueEvolution == false
                emission = Emission(0, 0, 0, 0,  false, false)
            end
            # Test for the possibility of a g -> qqbar emission
            branchType  = 3
            for flavor in range(1, 5)
                emissionTemp = generateEmissions(Q, Qcut, aSover, branchType, masses)

                # Accept the emission if the generated t value is greater 
                if emissionTemp.continueEvolution == true && emissionTemp.t > emission.t
                    emission = emissionTemp
                    pb.m = mufunc(pa.m, Qg(pa.m))
                    pc.m = mufunc(pa.m, Qg(pa.m))
                    pb.id = flavor
                    pc.id = -flavor
                end
            end

        end
        # Rescale the evolution variable
        Q = sqrt(emission.t)
        # Condition to break out of the loop: 
        # either an emission has been generated or the evolution is terminated
        if emission.Generated == true || emission.ContinueEvolution == false
            break
        end

    end


    if emission.continueEvolution == false
        pa.continueEvolution = false
        pa.status = 1 # Set the parent particle to final state if there is not emission
        pa.virtuality = 0 # The virtuality is zero if the particle has no emission
        return
    end

    pb.t = emission.t
    pc.t = emission.t
    pb.z = emission.z
    pc.z = 1 - emission.z 

    pb.alpha = pa.alpha * pb.z
    pc.alpha = pa.alpha * pc.z

    pb.phi = emission.phi 
    pc.phi = emission.phi

    pT = sqrt(emisison.pTsq)
    vmsq = getVirtMsq(emission.t, emission.z) # Get the virtuality of the emitting particle

    pa.virtuality = vmsq

    pa.status = -1 # Set the emitting particle to initial state for emission

    pb.pT = pT
    pc.pT = pT 

    pb.status = 1
    pc.status = 1
    pb.continueEvolution = true
    pc.continueEvolution = true 

    return
end


function showerParticle(jet::Jet, particle::Particle, Qmin::Float64, aSover::Float64)
    # Append the progenitor to the jet's list of particles
    push!(jet.AllParticles, particle)

    i = 1

    while i < 26
        # Check if the current index is out of bounds for the list of particles in the jet
        # If so, we are done with showering this jet
        if i > length(jet.AllParticles) -1
            return
        end
        pa = jet.AllParticles[i]
        # Get the child particles' templates
        pb = Particle(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, [], 1, 1, 0, 0, true, [particle], [])
        pc = Particle(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, [], 1, 1, 0, 0, true, [particle], [])
        evolveParticle(pa, pb, pc, Qmin, aSover)
        # If the evolution is terminated, end this branch
        if pa.continueEvolution == false
            i += 1
            continue
        end
        # If not then set the current index to the b particle and append the c particle to the list
        jet.AllParticles[i] = pb
        push!(jet.AllParticles, Pc)
        # Append the child particles, b and c, to the list of children in the parent particle
        push!(particle.children, [pb, pc])

    end

end


function showerEvent(event, Qmin::Float64, aSover::Float64)

    newEvent= Event([], [])
    plist = []
    jets = Jet[]
    # Go through the list of particles in the event and get the ones that will be showered and append them to a list
    # This is needed since both particles are needed for the initial evolution scale
    for p in event.Jets
        # Check if the particle is an electron or positron to skip it
        if abs(p.id) == 11
            push!(newEvent.AllParticles, p)
        elseif abs(p.id) < 6 && abs(p.id) > 0 && p.status == 1 # Only shower final state particles
            push!(plist, p)
        end 
    end

    # Set the initial evolution scale for the two progenitors
    plist[1].t, plist[2].t = EvolutionScale(plist[1], plist[2])

    for (i, p) in enumerate(plist)
        jet = Jet([], p)

        showerParticle(jet, p, Qmin, aSover)
        reconSudakovBasis(p, plist[end - (i - 1)])
        rotateMomentaLab(p, jet.AllParticles)

        push!(jets, jet)
    end

    newEvent.AllParticles = globalMomCons(newEvent.AllParticles, jets)

    return newEvent
end
