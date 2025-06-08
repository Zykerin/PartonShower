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


# Define the E(t) or Emission scale function
def E(t, Q, Rp, aSover, t0, tGamma):
    
    zup, zlow = zbounds(t, t0)
    
    r =  tGamma(zup, aSover) - tGamma(zlow, aSover)
    return np.log(t / Q**2) -  (1 /r) * np.log(Rp)



# Define the function to determine the t value
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
    Ems= emissioninfo(0,0, 0, 0, 0, True, True)
    # Get the t emission value
    #print(Q, Qcut, R1, aSover, tfac)
    match branch_type:
        case 1:
            Ems.t, Ems.ContinueEvolve = tEmission(Q, Qcut, R1, aSover, tfac, tGamma_gg)
        case 2:
            Ems.t, Ems.ContinueEvolve = tEmission(Q, t0, R1, aSover, tfac, tGamma_qq)
        case 3:
            Ems.t, Ems.ContinueEvolve = tEmission(Q, Qcut, R1, aSover, tfac, tGamma_gq)
        case 4:
            Ems.t, Ems.ContinueEvolve = tEmission(Q, Qcut, R1, aSover, tfac, tGamma_qg)
    # Determine if there was a generated t emission
    
    if Ems.ContinueEvolve == False:
        
        Ems.z = 1
        Ems.pTsq = 0
        Ems.Vmsq = 0
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

        # Terminate the evolution if a requirement is met.
        # This then stops this branch's evolution
        if Emission.ContinueEvolve == False:
            # Add the magnitude of the quark with respect to its origina direction
            ps.append(Particle(21, np.sqrt(Emission.t), Emission.z, 0, 0, 0, pmag, 0, pmag, 0, False))
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
                p = Particle(21, np.sqrt(Emission.t), Emission.z, Pt, Pt * np.cos(Emission.phi), Pt * np.sin(Emission.phi), (1 -Emission.z) * pmag, 0, Ei, Emission.phi, Emission.ContinueEvolve)
            elif branch_type == 3:
                p = Particle(-3, np.sqrt(Emission.t), Emission.z, Pt, Pt * np.cos(Emission.phi), Pt * np.sin(Emission.phi), (1 -Emission.z) * pmag, 0, Ei, Emission.phi, Emission.ContinueEvolve)
           
            # Resacale the magnitude of the momnetum with the z emission
            pmag = Emission.z * pmag
            ps.append(p)
    # Add the magnitude of the quark with respect to its origina direction.
    ps.append(Particle(21, np.sqrt(Emission.t), Emission.z, 0, 0, 0, pmag, 0, pmag, 0, False))
    return ps
'''

# NEw function to evolve a particle with the competition.
# I figured it would be better to create a seperate function so I can switch back to the old one for testing.
# The function to evolve a certain particle
def Evolve(Pa, Pb, Pc, Qc, aSover):
    # Cases: 1: g -> gg, 2: q -> qg, 3: g -> qqbar, 4: q -> gq
    branch_type = 4
    momenta = []
    t0 = Qc**2
    tscalcuttoff = 4
    # Set the initial emission data class values
    Emission = emissioninfo(Pa.t**2, 1, 0, 0, True, True)
    # Loop until the evolution variable reaches the cuttoff 
    if np.sqrt(Pa.t_at_em) * Pa.z_at_em > np.sqrt( tscalcuttoff):
        # Code to determine which branch is in effect. WIP
        if Pa.typ <= 6 and Pa.typ >= -6:
            branch_type = 2
            Emission = GenerateEmissions(np.sqrt(Emission.t) * Emission.z, t0, aSover, branch_type)
        elif Pa.typ == 21:
            branch_type = 1
            Emission2 = GenerateEmissions(np.sqrt(Emission.t) * Emission.z, t0, aSover, branch_type)
            for flavor in range(6):
                Emission_temp = GenerateEmissions(np.sqrt(Emission.t) * Emission.z, t0, aSover, branch_type)
                if Emission_temp.t > Emission.t:
                    branch_type = 3
                    Pb.typ = flavor
                    Pc.typ = -flavor
                    Emission = Emission_temp
                    continue
                else:
                    Emission = Emission2
            
        else:
            print('ERROR: Invalid Particle type')
        # Get the values.
        # Terminate the evolution if a requirement is met.
        # This then stops this branch's evolution
        if Emission.ContinueEvolve == False:
            Pa.ContinueEvolution = False
            return
        
        
        
        Pb.t_at_em = Emission.t
        Pc.t_at_em = Emission.t
        Pb.z_at_em = Emission.z
        Pc.z_at_em = 1- Emission.z
        
        Pt = np.sqrt(Emission.Ptsq)
        pmag = np.sqrt(Pa.px**2 + Pa.py**2 + Pa.pz**2)
        
        Pb.px = Pt * np.cos(Emission.phi)
        Pb.py = Pt * np.sin(Emission.phi)
        Pb.pz = Emission.z_at_em * pmag
        Pb.E = np.sqrt(Pb.px**2 + Pb.py**2 + Pb.pz**2)
        
        Pc.px = -Pt * np.cos(Emission.phi)
        Pc.py = -Pt * np.sin(Emission.phi)
        Pc.pz = (1- Emission.z_at_em) * pmag
        Pc.E = np.sqrt(Pc.px**2 + Pc.py**2 + Pc.pz**2)
        
        Pb.m = 0
        Pc.m = 0
        
        Rotated_particles = rotate_momenta(Pa, [Pb, Pc])
        Pb = Rotated_particles[0]
        Pc = Rotated_particles[1]
        
        # Append the emissions and then continue
        if Emission.z != 1:
            momenta.append([Pb.typ,])
            emissions.append( (np.sqrt(Emission.t), Emission.z, np.sqrt(Emission.Ptsq), np.sqrt(Emission.Vmsq)))
        
    return 

def Shower_particle(particle, Qmin, aSover):
    
    #for i in range(25):
    Pa = particle
    Pb = Particle(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, True)
    Pc = Particle(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, True)
    Evolve(Pa, Pb, Pc, Qmin, aSover)
    if Pa.ContinueEvolution == False:
        return Pb, Pc
    Jet.Particles.append(Pb)
    Jet.Particles.append(Pc)
    Shower_particle(Jet, Pb, Qmin, aSover)
    Shower_particle(Jet, Pc, Qmin, aSover)
    
    
    return
'''
# Define the function to shower the events
def Shower_Events(Event, Qmin, aSover):
    # List of jets and particles
    Jets = []
    AllParticles = []
    '''
    for p in Event:
        ps = Shower_particle()
    '''
    
    
    # Go through the first event and shower each particle
    # This was for testing/comparison
    for i in Events[0]:
        # Test to only evolve quarks
        if abs(i.typ) == 11:
            AllParticles.append(i)
            pass
        else:
            ps = Evolve(i, Qc, aSover )
            
            # Rotate the particle with the lab
            rotated = rotate_momenta_lab(i, ps)
            AllParticles.extend(rotated)
            #AllParticles.append(rotated)
            jet = Jet(i, rotated)
            Jets.append(jet)
    showered_particles = Glob_mom_cons(AllParticles, Jets)

    return showered_particles, Jets


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
    
    newevent = []
    # Go throught each particle in the event and create a respective particle data class
    for p in event:
        P = Particle(p[0], 0, 0, np.sqrt(p[2]**2 + p[3]**2 + p[4]**2), p[2], p[3], p[4], p[6], p[5], 0)
        newevent.append(P)
    Events.append(newevent)


ShoweredParticles, ShoweredJets = Shower_Events(Events, Qc, aSover)
    

#print(check_mom_cons(showered_particles))
sys.exit()
# Old test for proper generation of splitting functions
Pa = Particle(1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
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

Y = np.array(dist) *(1- X)
testP = Pqq(X) *(1- X) 

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