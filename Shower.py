import numpy as np
import random as rand
import scipy.optimize 
from Classes import * # Import the classes
from Constants import * # Import the constants
from SplittingFunctions import * # Import the splitting functions
from Kinematics import * # Import the kinematic functions
from alphaS import * # ALphaS coupling constant
import math
import sys
rand.seed(12345)



# Function to determine the starting evolution scale
def EvolutionScale(p1, p2):
    
    match EvolveType :
        case 'Old':
            return [p1.E**2, p2.E**2]
        case 'QTilde':
            Q2 = (0.5 * p1.E + 0.5* p2.E)**2 - (p1.Px + p2.Px)**2 - (p1.Py + p2.Py)**2 - (p1.Pz + p2.Pz)**2

            print(Q2, p1.E**2)
            #sys.exit()
            
            b = p1.m**2 / Q2
            c = p2.m**2/Q2
            lam = np.sqrt(1 + b**2 + c**2 - 2*b - 2*c - 2*b*c)
            ktildb = 0.5 * (1 + b - c + lam)
            ktildc = 0.5 * (1 - b + c + lam)
            
            Qtilde = [ np.sqrt(Q2 * ktildb), np.sqrt(Q2 * ktildc)]
            return Qtilde



# Function to get the true value of alphaS using the PDF alphaS
# This program is using the pT scale
def GetalphaS(t, z, Qcut): 
    scale = z * (1-z) * np.sqrt(t) # = pT of emission
    if scale < Qcut:
        return aS.alphasQ(Qcut)
    return aS.alphasQ(scale)

# The function to determine the alphaS overestimate
def GetalphaSOver(Qcut):
    return aS.alphasQ(Qcut) 


# Define the E(t) or Emission scale function
def E(t, Q, Rp, aSover, Qcut, tGamma, mu, branchType):
    
    #zup, zlow = zbounds(t, Qcut)
    zup, zlow = zBounds(mu, t, Qcut, branchType)
    r =  tGamma(zup, aSover) - tGamma(zlow, aSover)
    return np.log(t / Q**2) -  (1 /r) * np.log(Rp)



# Define the function to determine the t value
# This is doen by numerically solving the emission scale function
def tEmission(Q, Qcut, R2, aSover, tfac, tGamma, mu, branchType):
    prec = 1E-4 # Precision for the solution
    argsol = (Q, R2, aSover, Qcut, tGamma, mu,  branchType)

    ContinuedEvolve = True
    t = scipy.optimize.ridder(E, 3.99 * pT2min , Q**2, args = argsol, xtol= prec)
    
    # If a root is not found, stop the evolution for this branch
    if abs(E(t, Q, R2, aSover, Qcut, tGamma, mu, branchType)) > prec:
        t = Q**2
        ContinuedEvolve = False
    return t, ContinuedEvolve


# Function to determine the z emission
def zEmission (t, t0, Rp, aSover, tGamma, inversetGamma, mu, branchType):  
    #zup, zlow = zbounds(t, t0)
    # mu, t, t0, branchType
    zup, zlow = zBounds(mu, t, t0, branchType)
    
    z = inversetGamma( tGamma(zlow, aSover) + Rp * (tGamma(zup, aSover) - tGamma(zlow, aSover)), aSover )
    return z

# Define the function to generate the emissions
def GenerateEmissions (Q, Qcut, aSover, tfac, branchType, mu):

    # Generate the three randomn numbers to determine whether or not to accept the emission.
    R1 = rand.random()
    R2 = rand.random()
    R3 = rand.random()
    R4 = rand.random()
    
    
    # Get an empty emission object to hold emission data.
    Ems= emissioninfo(0,1, 0, 0, 0, True, True)

    # Get the t emission value
    match branchType:
        case 1:
            
            Ems.t, Ems.ContinueEvolve = tEmission(Q, Qcut, R1, aSover, tfac, tGamma_gg, mu, branchType)
            
        case 2:
            
            Ems.t, Ems.ContinueEvolve = tEmission(Q, Qcut, R1, aSover, tfac, tGamma_qq, mu, branchType)
            
        case 3:
            
            Ems.t, Ems.ContinueEvolve = tEmission(Q, Qcut, R1, aSover, tfac, tGamma_gq, mu, branchType)
            
        case 4:
            
            Ems.t, Ems.ContinueEvolve = tEmission(Q, Qcut, R1, aSover, tfac, tGamma_qg, mu, branchType)
        
        case _:
            raise Exception('Invalid Branching option.')
            
    # Determine if there was a generated t emission   
    if Ems.ContinueEvolve == False:   
        Ems.z = 1
        Ems.pTsq = 0
        Ems.Vmsq = 0
        #Ems.Generated = False
        return Ems
    
    
    # Get the z emissiom, tranverse momentum squared, and the virtual mass squared
    match branchType:
        case 1:
            
            Ems.z = zEmission(Ems.t, Qcut, R2, aSover, tGamma_gg, inversetGamma_gg, mu, branchType)
            
        case 2:
            
            Ems.z = zEmission(Ems.t, Qcut, R2, aSover, tGamma_qq, inversetGamma_qq, mu, branchType)
            
        case 3:
            
            Ems.z = zEmission(Ems.t, Qcut, R2, aSover, tGamma_gq, inversetGamma_gq, mu, branchType)
            
        case 4:
            
            Ems.z = zEmission(Ems.t, Qcut, R2, aSover, tGamma_qg, inversetGamma_qg, mu, branchType)
        
        case _:
            raise Exception('Invalid Branching option.')
    # Get the transverse momentum squared and virtual mass squared
    Ems.Ptsq = transversemmsq(Ems.t, Ems.z)
    Ems.Vmsq = virtualmass(Ems.t, Ems.z)

    # Determine whether the transverse momentum is physical
    if Ems.Ptsq < pT2min:
        print('Invalid transverse momentum')
        Ems.Generated = False
    

    # Determine whether or no to accept the t value, and inturn accept the emission
    match branchType:
        case 1:
            
            if R3 > Pgg(Ems.z) / Pgg_over(Ems.z):
                Ems.Generated = False
                
        case 2:
            
            if R3 > Pqq(Ems.z) / Pqq_over(Ems.z):
                Ems.Generated = False
                
        case 3:
            
            if R3 > Pgq(Ems.z) / Pgq_over(Ems.z):
                Ems.Generated = False
                
        case 4:
            
            if R3 > Pqg(Ems.z) / Pqg_over(Ems.z):
                Ems.Generated = False
                
        case _:
            raise Exception('Invalid Branching option.')
                
    # Compare the alphaS value and overestimate to a random number to veto according to it
    if R4 > GetalphaS(Ems.t, Ems.z, Qcut) / aSover:
        Ems.Generated = False 
        
    
    # If any of the tests are true, then there is no emission and return these values
    if Ems.Generated == False:
        Ems.z = 1
        Ems.Ptsq = 0
        Ems.Vmsq = 0  

    return Ems

# The function to evolve a certain particle
# This is the old one for testing
def Evolve(pa, pslist, Qc, aSover):
    
    # Set the branch type for now. This is meant so one can easily switch splitting functions for testing
    # Cases: 1: g -> gg, 2: q -> qg, 3: g -> qqbar, 4: q -> gq
    branch_type = 2
    
    
    fac_t = 3.999 # Minimum value for the cutoff to try emissions
    tscalcuttoff = 4 # ACtual cuttoff
    
    # The miminum evolution scale
    t_min = Qc**2
    
    # Set the initial emission data class values.
    # These will be written over later
    Emission = emissioninfo(pa.E**2, 1, 0, 0, 0, True, True)
    
    # List to hold the particles/emisisons
    ps = []
    
    # The magnitude of the momentum of the parent particle
    pmag = np.sqrt(pa.Px**2 + pa.Py**2 + pa.Pz**2)
    
    
    
    # Get the starting evolution scale
    Emission.t = EvolutionScale(pslist[0], pslist[1])[0]
    
    # Evolve the particle until the emission is past the cuttoff or another condition is met.
    while np.sqrt(Emission.t) * Emission.z > np.sqrt( tscalcuttoff * t_min) :
        
        Pb = Particle(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False)
        Pc = Particle(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False)
        
        masses = [mufunc(pa.m, [Pb, Pc]), Qg(pa.m)]
        
        # Get the emission values.
        Emission = GenerateEmissions(np.sqrt(Emission.t) * Emission.z, np.sqrt(t_min), aSover, fac_t, branch_type, masses)
        
        # Terminate the evolution if a requirement is met.
        # This then stops this branch's evolution
        if Emission.ContinueEvolve == False:
            # Add the magnitude of the quark with respect to its origina direction
            ps.append(Particle(pa.typ, 1, np.sqrt(Emission.t), Emission.z, 0, 0, 0, pmag, 0, pmag, 0, False))
            pa.ContinueEvolution = False
            return ps
        
        # Return emssions if the current one is past the cutt offf
        if Emission.t < tscalcuttoff * t_min:
            print('Hit Cuttoff, stopping Evolution')
            pa.ContinueEvolution = False
            return ps
        
        
        # Append the emissions and then continue
        if Emission.z != 1:
            
            Pt = np.sqrt(Emission.Ptsq) 
            Emission.phi = (2*rand.random() - 1)*np.pi # Generated phi value
            Ei = np.sqrt(( 1- Emission.z)**2 * pmag**2 + Pt**2)
            
            # Depending on the branch, append the appropiate particle
            if branch_type == 2:
                p = Particle(21, 1, np.sqrt(Emission.t), Emission.z, Pt, Pt * np.cos(Emission.phi), Pt * np.sin(Emission.phi), (1 -Emission.z) * pmag, 0, Ei, Emission.phi, Emission.ContinueEvolve)
            elif branch_type == 3:
                p = Particle(-3, 1, np.sqrt(Emission.t), Emission.z, Pt, Pt * np.cos(Emission.phi), Pt * np.sin(Emission.phi), (1 -Emission.z) * pmag, 0, Ei, Emission.phi, Emission.ContinueEvolve)

            # Resacale the magnitude of the momnetum with the z emission
            pmag = Emission.z * pmag
            ps.append(p)
    
    # Add the magnitude of the quark with respect to its origina direction.
    ps.append(Particle(pa.typ, 1, np.sqrt(Emission.t), Emission.z, 0, 0, 0, pmag, 0, pmag, 0, False))
    return ps

# New function to evolve a particle with the competition.
# I figured it would be better to create a seperate function so I can switch back to the old one for testing.
# The function to evolve a certain particle
def EvolveParticle(Pa, Pb, Pc, Qc, aSover):
    tMin = Qc**2
    fac_t = 3.999 # Minimum value for the cutoff to try emissions
    tscalcuttoff = 4# ACtual cuttoff
    # Set the initial emission data class values
    Emission = emissioninfo(Pa.t_at_em, 1, 0, 0, 0, True, True)
    
    # Set the evolution variable from the parent particle's information
    Q = np.sqrt(Pa.t_at_em) * Pa.z_at_em
    
    # Loop until either the particle evolution is terminated or an emission is generated
    # This is an implementation of a do while loop from another codebase 
    while True:
        # If the evolution variables is below the cutoff, 
        if Q < np.sqrt(tscalcuttoff * Qc**2):
            Pa.ContinueEvolution = False
            return
        
        # Code to determine which branch is in effect. 
        # If the parent particle is a quark, then the outgoing quark is the same type
        # For now there is only q -> qg for initial quark
        if abs(Pa.typ) < 6 and abs(Pa.typ) > 0:
            branch_type = 2
            Emission = GenerateEmissions(Q, np.sqrt(tMin), aSover, fac_t, branch_type)
            Pb.typ = Pa.typ
            Pc.typ = 21
            
        # For gluons, there are g -> gg and g -> qqbar.
        # So a competition for which to accept is needed.
        elif abs(Pa.typ) == 21:
            # Get emission for g -> gg
            branch_type = 1
            Emission = GenerateEmissions(Q, np.sqrt(tMin), aSover, fac_t, branch_type)
            
            # Set the child particles type
            Pb.typ = 21
            Pc.typ = 21
            
            # If stopped evolution, set t to 0 and continue for test of g -> qqbar emision
            if Emission.ContinueEvolve == False:
                
                Emission.t = 0
            
            # Go through each of the quark flavors and test whether one is emitted and the t emission is greater than the g -> gg t emission
            for flavor in range(1, 6):
                # Get emission for g -> qqbar
                branch_type = 3
                EmissionTemp = GenerateEmissions(Q, np.sqrt(tMin), aSover, fac_t, branch_type)
                
                # Accept g -> qqbar emission if true.
                if EmissionTemp.ContinueEvolve == True and EmissionTemp.t > Emission.t:
                    
                    Pb.typ = flavor
                    Pc.typ = -flavor
                    Emission = EmissionTemp
        else:
            print('ERROR: Invalid Particle type')

        Q = np.sqrt(Emission.t) 
        if Emission.Generated == True or Emission.ContinueEvolve == False:
            break
    # Terminate the evolution if needed.
    # This then stops this branch's evolution
    if Emission.ContinueEvolve == False:
        Pa.ContinueEvolution = False
        return
    # Generated the phi value for this emission
    Emission.phi = (2*rand.random() - 1)*np.pi 
    
    # Set the children particles' t and z emission values
    Pb.t_at_em = (Emission.t)
    Pc.t_at_em = (Emission.t)
    Pb.z_at_em = Emission.z
    Pc.z_at_em = 1- Emission.z
    
    # Get transverse momentum and momentum magnitude of mother particle
    Pt = np.sqrt(Emission.Ptsq)
    pmag = np.sqrt(Pa.Px**2 + Pa.Py**2 + Pa.Pz**2)
    
    # Set the b child particle's momentum and energy values.
    Pb.Px = Pt * np.cos(Emission.phi)
    Pb.Py = Pt * np.sin(Emission.phi)
    Pb.Pz = Emission.z* pmag
    Pb.E = np.sqrt(Pb.Px**2 + Pb.Py**2 + Pb.Pz**2)
    
    # Set the c child particle's momentum and energy values.
    Pc.Px = -Pt * np.cos(Emission.phi)
    Pc.Py = -Pt * np.sin(Emission.phi)
    Pc.Pz = (1- Emission.z) * pmag
    Pc.E = np.sqrt(Pc.Px**2 + Pc.Py**2 + Pc.Pz**2)
    
    # Set the other variables for the particles
    Pb.status = 1
    Pc.status = 1
    Pb.Pt = Pt
    Pc.Pt = Pt
    Pb.ContinueEvolution = True
    Pc.ContinueEvolution = True
    Pb.m = 0
    Pc.m = 0
    
    # Rotate the child particles to allign with the parent particle
    RotatedParticles = RotateMomentaLab(Pa, [Pb, Pc])
    Pb = RotatedParticles[0]
    Pc = RotatedParticles[1]
    
    return 

def ShowerParticle(jet, particle, Qmin, aSover):
    # Append the parent particle to jet's particles
    jet.Particles.append(particle)
    
    i =0
    
    while i < 25:
        # Check if the index is out of bounds for the list
        if i > len(jet.Particles) - 1:
            return
        Pa = jet.Particles[i]
        # Get children particles' template
        Pb = Particle(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False)
        Pc = Particle(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False)
        EvolveParticle(Pa, Pb, Pc, Qmin, aSover)

        if Pa.ContinueEvolution == False:
            i += 1
            continue
        jet.Particles[i] = Pb
        jet.Particles.append(Pc)
    
    # Recursive soltion. Doesn't work for some reason
    '''
    Pa = (particle)
    # Get children particles' template
    Pb = Particle(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False)
    Pc = Particle(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False)
    # Evolve the particles
    EvolveParticle(Pa, Pb, Pc, Qmin, aSover)
    # Escape if evolution is terminated
    if Pa.ContinueEvolution == False:
        return
    else:
    # Else append the children particles to the jet and shower them recursively
        ShowerParticle(jet, Pb, Qmin, aSover)
        ShowerParticle(jet, Pc, Qmin, aSover)
    '''
    return

def ShowerEvent(event, Qmin, aSover):
    
    NewEvent = Event([],[])

    # Go through the list of progenitors in the event and shower each
    for p in event.Jets:
            jet = Jet(p, [])
            # Test if particle is electron or positron and then append it
            if abs(p.typ) == 11:
                NewEvent.AllParticles.append(p)
                continue
            # Shower the progenitor
            ShowerParticle(jet, p, Qmin, aSover)   
            
            #jet = Jet(p, Rotated_particles)
            NewEvent.Jets.append(jet)
            
    
    # Apply global momentum for this event
    NewEvent.AllParticles = Glob_mom_cons(NewEvent.AllParticles, NewEvent.Jets)
    
    return NewEvent

# This is for testing as this does not involve the new Evolution function with competition
# Define the testing function to shower the events
def Shower_Evens(Event, Qmin, aSover):
    
    AllParticles = []
    Jets = []
    pslist = []
    # Go through the first event and shower each particle
    # This was for testing/comparison
    for i in Event.Jets:
        # Test to only evolve quarks
        if abs(i.typ) == 11:
            AllParticles.append(i)
            continue
        elif abs(i.typ) < 6 and abs(i.typ) > 0 and i.status==1:
            pslist.append(i)
            
    for i in pslist:
            ps = Evolve(i, pslist, Qc, aSover )
            
            # Rotate the particle with the lab
            rotated = RotateMomentaLab(i, ps)
            AllParticles.extend(rotated)
            #AllParticles.append(rotated)
            jet = Jet(i, rotated)
            Jets.append(jet)
    AllParticles = Glob_mom_cons(AllParticles, Jets)

    return AllParticles, Jets

