import random as rand
from alphaS import * # ALphaS coupling constant
import numpy as  np
import scipy.optimize 
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
from LHEWriter import * # part to write to a lhe file
from LHEReader import * # part to read the lhe file
from SplittingFunctions import * # Splitting functions file
from Kinematics import * # Kinematics reconstuction file
from Classes import * # File containing data classes
import math
# The cuttoff scale
Q = 1000 # The Hard scale? - the scale attributed to the hard process 
aS= 0.118 # The coupling constant
aSover = aS # Use the fixed overestimate alphaS constant
# The QCD constants
Nc = 3
Cf = (Nc**2 -1)/ (2 * Nc) # The quark color factor
Ca = Nc
Tr = 1/2

Qc = 0.935

rand.seed(12345)
# Define the E(t) or Emission scale function
def E(t, Q, Rp, aSover, t0, tGamma):
    
    zup, zlow = zbounds(t, t0)
    
    r =  tGamma(zup, aSover) - tGamma(zlow, aSover)
    return np.log(t / Q**2) -  (1 /r) * np.log(Rp)



# Define the function to determine the t value
# This is doen by numerically solving the emission scale function
def tEmission(Q, Qc, R2, aSover, tfac, tGamma):
    prec = 1E-4 # Precision for the solution
    argsol = (Q, R2, aSover, Qc, tGamma)

    ContinuedEvolve = True
    t = scipy.optimize.ridder(E, tfac * Qc**2, Q**2, args = argsol, xtol= prec)
    
    # If a root is not found, stop the evolution for this branch
    if abs(E(t, Q, R2, aSover, t0, tGamma)) > prec:
        t = Q**2
        ContinuedEvolve = False
    return t, ContinuedEvolve


# Function to determine the z emission
def zEmission (t, t0, Rp, aSover, tGamma, inversetGamma):  
    zup, zlow = zbounds(t, t0)
    z = inversetGamma( tGamma(zlow, aSover) + Rp * (tGamma(zup, aSover) - tGamma(zlow, aSover)), aSover )
    return z

# Define the function to generate the emissions
def GenerateEmissions (Q, Qcut, aSover, tfac, branch_type):

    # Generate the three randomn numbers to determine whether or not to accept the emission.
    R1 = rand.random()
    R2 = rand.random()
    R3 = rand.random()
    # Get an empty emission object to hold emission data.
    Ems= emissioninfo(0,1, 0, 0, 0, True, True)
    
    #print(Q, Qcut, R1, aSover, tfac)
    # Get the t emission value
    match branch_type:
        case 1:
            
            Ems.t, Ems.ContinueEvolve = tEmission(Q, Qcut, R1, aSover, tfac, tGamma_gg)
            
        case 2:
            
            Ems.t, Ems.ContinueEvolve = tEmission(Q, Qcut, R1, aSover, tfac, tGamma_qq)
            
        case 3:
            
            Ems.t, Ems.ContinueEvolve = tEmission(Q, Qcut, R1, aSover, tfac, tGamma_gq)
            
        case 4:
            
            Ems.t, Ems.ContinueEvolve = tEmission(Q, Qcut, R1, aSover, tfac, tGamma_qg)
            
            
    # Determine if there was a generated t emission   
    if Ems.ContinueEvolve == False:   
        Ems.z = 1
        Ems.pTsq = 0
        Ems.Vmsq = 0
        #Ems.Generated = False
        return Ems
    
    
    # Get the z emissiom, tranverse momentum squared, and the virtual mass squared
    match branch_type:
        case 1:
            
            Ems.z = zEmission(Ems.t, Qcut, R2, aSover, tGamma_gg, inversetGamma_gg)
            
        case 2:
            
            Ems.z = zEmission(Ems.t, Qcut, R2, aSover, tGamma_qq, inversetGamma_qq)
            
        case 3:
            
            Ems.z = zEmission(Ems.t, Qcut, R2, aSover, tGamma_gq, inversetGamma_gq)
            
        case 4:
            
            Ems.z = zEmission(Ems.t, Qcut, R2, aSover, tGamma_qg, inversetGamma_qg)
            
            
    # Get the transverse momentum squared and virtual mass squared
    Ems.Ptsq = transversemmsq(Ems.t, Ems.z)
    Ems.Vmsq = virtualmass(Ems.t, Ems.z)

    # Determine whether the transverse momentum is physical
    if Ems.Ptsq < 0:
        print('Invalid transverse momentum')
        Ems.Generated = False
    

    # Determine whether or no to accept the t value, and inturn accept the emission
    match branch_type:
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
         
    # If any of the tests are true, then there is no emission and return these values
    if Ems.Generated == False:
        Ems.z = 1
        Ems.Ptsq = 0
        Ems.Vmsq = 0  

    return Ems

# The function to evolve a certain particle
# This is the old one for testing
def Evolve(pa, Qc, aSover):
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
    ps = []
    # The magnitude of the momentum of the parent particle
    pmag = np.sqrt(pa.Px**2 + pa.Py**2 + pa.Pz**2)
    # Evolve the particle until the emission is past the cuttoff or another condition is met.
    while np.sqrt(Emission.t) * Emission.z > np.sqrt( tscalcuttoff * t_min) :
        # Get the emission values.
        
        Emission = GenerateEmissions(np.sqrt(Emission.t) * Emission.z, np.sqrt(t_min), aSover, fac_t, branch_type)
        #print(Emission.t, Emission.z)
        # Terminate the evolution if a requirement is met.
        # This then stops this branch's evolution
        if Emission.ContinueEvolve == False:
            # Add the magnitude of the quark with respect to its origina direction
            ps.append(Particle(21, 1, np.sqrt(Emission.t), Emission.z, 0, 0, 0, pmag, 0, pmag, 0, False))
            pa.ContinueEvolution = False
            return ps
        
        # Return emssions if the current one is past the cutt offf
        if Emission.t < tscalcuttoff * t_min:
            pa.ContinueEvolution = False
            return ps
        # Append the emissions and then continue
        if Emission.z != 1:
            Pt = np.sqrt(Emission.Ptsq) 
            Emission.phi = (2*rand.random() - 1)*np.pi # Generated phi value
            Ei = np.sqrt(( 1- Emission.z)**2 * pmag**2 + Pt**2)
            # Depending on the branch, append the appropiate particle
            if branch_type == 1 or 2:
                p = Particle(21, 1, np.sqrt(Emission.t), Emission.z, Pt, Pt * np.cos(Emission.phi), Pt * np.sin(Emission.phi), (1 -Emission.z) * pmag, 0, Ei, Emission.phi, Emission.ContinueEvolve)
            elif branch_type == 3:
                p = Particle(-3, 1, np.sqrt(Emission.t), Emission.z, Pt, Pt * np.cos(Emission.phi), Pt * np.sin(Emission.phi), (1 -Emission.z) * pmag, 0, Ei, Emission.phi, Emission.ContinueEvolve)
           
            # Resacale the magnitude of the momnetum with the z emission
            pmag = Emission.z * pmag
            ps.append(p)
    # Add the magnitude of the quark with respect to its origina direction.
    ps.append(Particle(21, np.sqrt(Emission.t), Emission.z, 0, 0, 0, pmag, 0, pmag, 0, False))
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
    
    # Loop until the evolution variable reaches the cuttoff 
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
    Emission.phi = (2*rand.random() - 1)*np.pi # Generated phi value
    
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
    
    Pb.status = 1
    Pc.status = 1
    
    Pb.Pt = Pt
    Pc.Pt = Pt
    
    Pb.ContinueEvolution = True
    Pc.ContinueEvolution = True
    
    # Set the c child particle's momentum and energy values.
    Pc.Px = -Pt * np.cos(Emission.phi)
    Pc.Py = -Pt * np.sin(Emission.phi)
    Pc.Pz = (1- Emission.z) * pmag
    Pc.E = np.sqrt(Pc.Px**2 + Pc.Py**2 + Pc.Pz**2)
    
    # Set the mass of each particle to 0 for now.
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
# Define the function to shower the events
def Shower_Evens(Event, Qmin, aSover):
    
    AllParticles = []
    Jets = []
    # Go through the first event and shower each particle
    # This was for testing/comparison
    for i in Event[0].Jets:
        # Test to only evolve quarks
        if abs(i.typ) == 11:
            AllParticles.append(i)
            pass
        else:
            ps = Evolve(i, Qc, aSover )
            
            # Rotate the particle with the lab
            rotated = RotateMomentaLab(i, ps)
            AllParticles.extend(rotated)
            #AllParticles.append(rotated)
            jet = Jet(i, rotated)
            Jets.append(jet)
    AllParticles = Glob_mom_cons(AllParticles, Jets)

    return AllParticles, Jets


# Get the input file, read and then parse it
inputfile = 'eejj_ECM206.lhe.gz'

events, weights, multiweights = readlhefile(inputfile)

emissions = [] 
nbins = 30 # The number of bins
Nevolve = 10000 # The number of evolutions

# Make the events in a for that works with my program. Dumb I know
Events = []
# Go through each event and parse through it
for event in events:
    
    newevent = Event([], [])
    # Go throught each particle in the event and create a respective particle data class
    for p in event:
        P = Particle(p[0], 1, p[5]**2, 1, np.sqrt(p[2]**2 + p[3]**2 + p[4]**2), p[2], p[3], p[4], p[6], p[5], 0, True)
        newevent.Jets.append(P)
    Events.append(newevent)


ShoweredEvents = []
for event in tqdm(Events):
    
    Ev = ShowerEvent(event, Qc, aSover)
    ShoweredEvents.append(Ev)


#ShoweredParticles, ShoweredJets = Shower_Evens(Events, Qc, aSover)
sys.exit()
t = []

for p in Events[3].Jets:
    if abs(p.typ) == 1:
        t.append(p)
        continue
    ps = Evolve(p, Qc, aSover)
    t.append(ps)

sys.exit()
#print(check_mom_cons(showered_particles))
# Old test for proper generation of splitting functions
Pa = Particle(1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
Pa.E = Q
for i in tqdm(list(range(Nevolve))):
   ps = Evolve(Pa, Qc, aSover)
   emissions = emissions + ps
# Set the arrays to obtain the physicals
ts = []
zs = []
Pt = []
Vm = []

# Get all the physicals
for particle in emissions:
    ts.append(particle.t_at_em)
    zs.append(particle.z_at_em)
    Pt.append(particle.Pt)
    Vm.append(0)



# Get the histogram bins and edges of the z values
dist, edges = np.histogram(zs, nbins, density = True)

X =[]
# Get the z values into an array from the edges
for i in range(len(edges)-1):
    X.append((edges[i+1] + edges[i])/2)

# Set the z values and bins arrays into a numpy array for easier use
X = np.array(X)

Y = np.array(dist)
testP = Pgq(X) 

# Set the constant to normalize the bins array to the comparison array to easily compare the two
integ = np.linalg.norm(testP)
norm = np.linalg.norm(Y)
Y = integ * Y/ norm

#plt.hist(Pt, density=True)
#plt.show()
#sys.exit()

plt.plot(X, Y, label='generated', lw= 0, marker='o', markersize=4, color = 'blue')
plt.plot(X, testP, label='analytical', color = 'red')
plt.minorticks_on()

plt.xlabel(r'$z$')
plt.legend(loc='upper left')

# Y label and title for g -> qqbar
#plt.ylabel(r'$P_{gq}(z)$')
#plt.title(r'$g \rightarrow q\bar{q}$ Splitting Function')

#Y label and title for g -> gg
#plt.ylabel(r'$P_{gg}(z)z(1-z)$')
#plt.title(r'$g \rightarrow gg $ Splitting Function')

# Y label and title for q -> qg 
plt.ylabel(r'$P_{qq}(z)(1-z)$')
plt.title(r'$q \rightarrow qg$ Splitting Function')

# y label and title for q -> gq
#plt.ylabel(r'$P_{qg}(z)z$')
#plt.title(r'$q \rightarrow gq$ Splitting Function')

plt.show()