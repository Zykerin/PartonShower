import random as rand
from alphaS import *
import numpy as  np
import scipy.optimize 
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R
import sys
from LHEWriter import * # part to write to a lhe file
from LHEReader import * # part to read the lhe file



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

t0 = 1 # The cuttoff scale
Q = 1000 # The Hard scale? - the scale attributed to the hard process 
aS= 0.118 # The coupling constant
aSover = aS # Use the fixed overestimate alphaS constant
# The QCD constants
Nc = 3
Cf = (Nc**2 -1)/ (2 * Nc) # The quark color factor
Ca = Nc
Tr = 1/2

# The g - > gg splitting function, overestimate, integral, and integral's inverse
def Pgg(z):
    return  Ca * ((1 - z * (1-z))**2) / (z * (1-z))
    #return   Ca * (z / (1.0 - z) + (1.0 - z) / z + z * (1.0 - z))
def Pgg_over(z):
    return Ca * (1/(1-z) + 1/z)
def tGamma_gg(z, aSover):
    return -Ca * (aSover/ (2 * np.pi)) * np.log(1/z - 1)
def inversetGamma_gg(z, aSover):
    return 1 / (1 + np.exp(-2 * np.pi * z/ (Ca * aSover)))


# The g -> qqbar splitting function
def Pgq(z):
    return Tr *(1- 2 * z * (1-z))
# The g -> qqbar splitting function overestimate
def Pgq_over(z):
    return Tr
# The g -> qqbar splitting function integral
def tGamma_gq(z, aSover):
    return Tr * (aSover/ (2 * np.pi)) * z
# The g -> qqbar splitting function integral inverse
def inversetGamma_gq(z, aSover):
    return (z * 2 * np.pi / (Tr * (aSover))) 


# The q -> gq splitting function, overestimate, integral, and inverse integral
def Pqg(z):
    return Pqq(1-z)
def Pqg_over(z):
    return Pqq_over(1-z)
def tGamma_qg(z, aSover):
    return 2 * Cf * (aSover/ (2 * np.pi)) * np.log(z)
def inversetGamma_qg(z, aSover):
    return np.exp(z / ( 2 * Cf * (aSover/ (2 * np.pi))))


# Define the q -> qg splitting function
def Pqq (z):
    return Cf * (  (1 + z**2) / (1 -z) )
# Define the overestimate of the p -> pq splitting function
def Pqq_over (z):
    return Cf * ( 2 / (1 - z))
# Define the analytical solution to the overestimate of t * Gamma integrated over z.
# For this program, this can be seen as rho tilda...
def tGamma_qq (z, aSover):
    return -2 * Cf * (aSover/ (2 * np.pi)) * np.log(1 - z)
# Define the inverse gamma function
def inversetGamma_qq (z, aSover):
    return 1 - np.exp(-z / ( 2 * Cf * (aSover/ (2 * np.pi))))


# Define the transverse momnentum sqaured
def transversemmsq (t, z):
    return z**2 * (1 - z) **2 * t

# Functuon to find the virtual mass squared.t
def virtualmass (t, z):
    return z * (1 - z ) * t
# Define the upper and lower bounds of z, not the overestimate scale versiom
def zbounds (t, t0):
    return 1- np.sqrt( t0/t), np.sqrt(t0/t)

# Get the rotation matrix
def rotationMatrix(v1, v2):
    k = np.cross(v1, v2)
    # Test if the norm of the cross product is zero
    if np.linalg.norm(k) == 0:
        RotMat = R.from_matrix([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]])   
    else:
        k = k / np.linalg.norm(k)
        # Form the rotation matrix from the scipy.spatial.transform rotation library
        # This is done for an easier rotation computation
        RotMat = R.from_matrix([[0, -k[2], k[1]],
                            [k[2], 0, -k[0]],
                            [-k[1], k[0], 0]])   
    
    return RotMat

# Function to check the momentum conservation
def check_mom_cons(Event):
    total = [0, 0, 0]
    # Go through the list of events
    for pm in Event:
        total[0] += pm.Px
        total[1] += pm.Py
        total[2] += pm.Pz
    
    return total

# Rotate the given particles to be along the z axis
def rotate_momenta(p, particles):
    
    rotated_particles = []
    
    pmag = np.sqrt(p.Px**2 + p.Py**2 + p.Pz**2)
    Matrix = rotationMatrix([0, 0, pmag], [p.Px, p.Py, p.Pz])
    for i in particles:
        v = [i.Px, i.Py, i.Pz]
        rotedVec = Matrix.apply(v)
        i.Px = rotedVec[0]
        i.Py = rotedVec[1]
        i.Pz = rotedVec[2]
        rotated_particles.append(i)
    return rotated_particles

# Get the boost factor beta for the outgouting jet (new) and parent jet (old)
def Boost_factor(k, new, old):
    
    qs = new[0]**2 + new[1]**2 + new[2]**2
    q = np.sqrt(qs)
    Q2 = new[3]**2 - new[0]**2 - new[1]**2 - new[2]**2
    kp = k* np.sqrt(old[0]**2 + old[1]**2 + old[2]**2)
    kps= kp**2
    betamag = (q * new[3] - kp * np.sqrt(kps + Q2)) / (kps + qs + Q2)
    
    beta = betamag * (k / kp) * np.array([old[0], old[1], old[2]])
    
    if betamag >= 0:
        return beta
    else:
        return [0, 0, 0]    

# The function to boost given particles with a given boost factor beta
def boost(particles, beta):
    
    bmag = np.sqrt(beta[0]**2 + beta[1]**2 + beta[2]**2)
    # Get the gamma factor
    gamma = 1 / np.sqrt(1 - bmag**2)
    
    boosted_particles = []
    # Go throught the entire list of particles and boost each one
    for p in particles:
    
        boosted_particle = p
    
        # Use the matrix of a lorentz boost from https://www.physicsforums.com/threads/general-matrix-representation-of-lorentz-boost.695941/
        boosted_particle.Px = (-gamma * beta[0] + (1 + (gamma - 1) * beta[0]**2 / bmag**2) + (gamma - 1) * beta[0] * beta[1] / bmag**2 + (gamma - 1) * beta[0] * beta[2] / bmag**2) * p.Px
        boosted_particle.Py = (- gamma * beta[1]  + (gamma - 1) * beta[1] * beta[0]/ bmag**2 + 1 + (gamma -1) * beta[1]**2 / bmag**2 + (gamma -1) * beta[1] * beta[2] / bmag**2) * p.Py
        boosted_particle.Pz = (-gamma * beta[2] + (gamma - 1) * beta[2] * beta[0] /bmag**2 + (gamma -1) * beta[2] * beta[1] / bmag**2 + 1 + (gamma -1) * beta[2]**2 / bmag**2) * p.Pz
        boosted_particle.E = (gamma - gamma * beta[0] - gamma * beta[1] - gamma * beta[2]) * p.E
         
        boosted_particles.append(boosted_particle)
        
    return boosted_particle

# The equation to numerically solve to find k
def K_eq(k, p, q, s):
    sump = 0
    # Loop through the momentum and sum up each term
    for i in range(len(p)):
        sump = sump + np.sqrt(k*k *p[i]**2 + q[i]**2)
        
    sump = sump - np.sqrt(s)
    
    return sump

# Solve for the k factor
def Solve_k_factor(pj, qj, s):
    
    kargs = [pj, qj, s]
    k = scipy.optimize.root(K_eq, 1.01, args = kargs)
    
    return k

# Define the function to perfom global momentum conservation on the jets and particles
def Glob_mom_cons(Jets):
    
    pj = [] # The momenta of the parent parton
    qj = [] # The momenta of the jets?
    # Initialize the total energy
    sqrts = 0
    
    newqs = []
    oldps = []
    
    # Iterate through the list of jets
    for jet in Jets:
        
        # Append the 3-momentum of the Jet's progenitor to the 
        pj.append([jet.Progenitor.Px**2 + jet.Progenitor.Py**2 + jet.Progenitor.Pz**2])
        
        oldps.append(jet.Progenitor.Px, jet.Progenitor.Py, jet.Progenitor.Pz, jet.Progenitor.E)
        # Add this jet's progenitor's energy
        sqrts += jet.Progenitor.E
        
        newq = [0, 0, 0, 0]
        # Iterate through the particles in the jet and add their momentum
        for p in jet.Particles:
            
            newq[0] = newq[0] + p.Px
            newq[1] = newq[1] + p.Py
            newq[2] = newq[2] + p.Pz
            newq[3] = newq[3] + p.E
        
        # Calculate the momentum and append it to the list of jet momentum
        qj2 = newq[3]**2 - newq[0]**2 - newq[1]**2 - newq[2]**2
        newqs.append(newq)
        qj.append(qj2)
        
    
    # Get the k factor
    k = Solve_k_factor(pj, qj, sqrts)
    
    # Iterate though the jets list and boost each
    for jet in Jets:
        
        rotated = rotate_momenta(jet.progenitor, jet.particles)
        
        
        beta = Boost_factor(k, newqs, oldps)
        
        boosted_particles = boost(rotated, beta)
        
    
    
    return boosted_particles

# Define the E(t) or Emission scale function
def E(t, Q, R, aSover, t0, tGamma):
    
    zup, zlow = zbounds(t, t0)
    
    r =  tGamma(zup, aSover) - tGamma(zlow, aSover)
    return np.log(t / Q**2) -  (1 /r) * np.log(R)



# Define the function to determine the t value
def tEmission(Q, t0, R, aSover, tGamma):
    prec = 1E-3 # Precision for the solution
    argsol = (Q, R, aSover, t0, tGamma)

    ContinuedEvolve = True
    t = scipy.optimize.ridder(E, 3.99, Q**2, args = argsol, xtol= prec)
    
    # If a root is not found, stop the evolution for this branch
    if abs(E(t, Q, R, aSover, t0, tGamma)) > prec:
        ContinuedEvolve = False
    return t, ContinuedEvolve



# Function to determine the z emission
def zEmission (t, t0, aSover, tGamma, inversetGamma):
    
    Rp = rand.random()
    zup, zlow = zbounds(t, t0)
    z = inversetGamma( tGamma(zlow, aSover) + Rp * (tGamma(zup, aSover) - tGamma(zlow, aSover)), aSover )

    return z



# Define the function to generate the emissions
def GenerateEmissions (Q, t0, aSover, branch_type):

    # Generate the three randomn numbers to determine whether or no to keep the solution
    R1 = rand.random()
    R2 = rand.random()
    
    Ems= emissioninfo(0,0, 0, 0, 0, True, True)
    
    # Get the t emission value
    match branch_type:
        case 1:
            Ems.t, Ems.ContinueEvolve = tEmission(Q, t0, R1, aSover, tGamma_gg)
        case 2:
            Ems.t, Ems.ContinueEvolve = tEmission(Q, t0, R1, aSover, tGamma_qq)
        case 3:
            Ems.t, Ems.ContinueEvolve = tEmission(Q, t0, R1, aSover, tGamma_gq)
        case 4:
            Ems.t, Ems.ContinueEvolve = tEmission(Q, t0, R1, aSover, tGamma_qg)
            
    # Determine if there was a generated t emission
    if Ems.ContinueEvolve == False:
        Ems.z = 1
        Ems.pTsq = 0
        Ems.Vmsq = 0
        return Ems
    
    # Get the z emissiom, tranverse momentum squared, and the virtual mass squared
    match branch_type:
        case 1:
            Ems.z = zEmission(Ems.t, t0, aSover, tGamma_gg, inversetGamma_gg)
        case 2:
            Ems.z = zEmission(Ems.t, t0, aSover, tGamma_qq, inversetGamma_qq)
        case 3:
            Ems.z = zEmission(Ems.t, t0, aSover, tGamma_gq, inversetGamma_gq)
        case 4:
            Ems.z = zEmission(Ems.t, t0, aSover, tGamma_qg, inversetGamma_qg)
    # Get the transverse momentum squared and virtual mass squared
    Ems.Ptsq = transversemmsq(Ems.t, Ems.z)
    Ems.Vmsq = virtualmass(Ems.t, Ems.z)
    # Determine whether the transverse momentum is physical
    if Ems.Ptsq < 0:
        Ems.Generated = False
    
    # Determine whether or no to accept the t value, and inturn accept the emission
    match branch_type:
        case 1:
            if R2 > Pgg(Ems.z) / Pgg_over(Ems.z):
                    Ems.Generated = False
        case 2:
            if R2 > Pqq(Ems.z) / Pqq_over(Ems.z):
                    Ems.Generated = False
        case 3:
            if R2 > Pgq(Ems.z) / Pgq_over(Ems.z):
                    Ems.Generated = False
        case 4:
            if R2 > Pqg(Ems.z) / Pqg_over(Ems.z):
                Ems.Generated = False
    Ems.phi = (2*rand.random() - 1)*np.pi
    # If any of the tests are true, then there is no emission and return these values
    if Ems.Generated == False:
        Ems.z = 1
        Ems.Ptsq = 0
        Ems.Vmsq = 0  
    
    return Ems


# The function to evolve a certain particle
def Evolve(pa, Qc, aSover):
    emissions = []
    momenta = []
    # Cases: 1: g -> gg, 2: q -> qg, 3: g -> qqbar, 4: q -> gq
    branch_type = 2
    
    t0 = Qc**2
    tscalcuttoff = 4
    # Set the initial emission data class values
    Emission = emissioninfo(pa.E**2, 1, 0, 0, 0, True, True)
    # Loop until the evolution variable reaches the cuttoff 
    ps = []
    # The magnitude of the momentum of the parent particle
    pmag = np.sqrt(pa.Px**2 + pa.Py**2 + pa.Pz**2)
    while np.sqrt(Emission.t) * Emission.z > np.sqrt( tscalcuttoff):
        # Get the values.
        
        Emission = GenerateEmissions(np.sqrt(Emission.t) * Emission.z, t0, aSover, branch_type)

        # Terminate the evolution if a requirement is met.
        # This then stops this branch's evolution
        if Emission.ContinueEvolve == False:
            return ps
        
        # Return emssions if the current one is out of bounds
        if Emission.t < ( tscalcuttoff):
            return ps
        # Append the emissions and then continue
        if Emission.z != 1:
            Pt = np.sqrt(Emission.Ptsq)
            
            
            if branch_type == 1 or 2:
                momenta.append([21, 1, Emission.Ptsq * np.cos(Emission.phi), Emission.Ptsq * np.sin(Emission.phi)])
            elif branch_type == 3:
                momenta.append([-3, 1, Emission.Ptsq* np.cos(Emission.phi), Emission.Ptsq * np.sin(Emission.phi)])
            emissions.append( [np.sqrt(Emission.t), Emission.z, np.sqrt(Emission.Ptsq), np.sqrt(Emission.Vmsq)])
            p = Particle(21, np.sqrt(Emission.t), Emission.z, np.sqrt(Emission.Ptsq), Pt * np.cos(Emission.phi), Pt * np.sin(Emission.phi), (1 -Emission.z) * pmag, 0, np.sqrt(( 1- Emission.z)**2 * pmag**2 + Pt), Emission.phi, Emission.ContinueEvolve)
            ps.append(p)
            
    
    return ps



'''
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
'''
def Shower_particle(Jet, particle, Qmin, aSover):
    
    #for i in range(25):
    Pa = particle
    Pb = Particle(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, True)
    Pc = Particle(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, True)
    Evolve(Pa, Pb, Pc, Qmin, aSover)
    if Pa.ContinueEvolution == False:
        return
    Jet.Particles.append(Pb)
    Jet.Particles.append(Pc)
    Shower_particle(Jet, Pb, Qmin, aSover)
    Shower_particle(Jet, Pc, Qmin, aSover)
    
    
    return

# Define the function to shower the events
def Shower_Events(Events, Qmin, aSover):
    
    
 #   for event in Events:
      #  Jets = []
      #  for i in range(2):
           # Jets.append(event[i])
    Pa = Particle(1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    Pa.E = Q**2
    
    jets = Jet(Pa, [])
    
    jet = Evolve(Pa, Qmin, aSover )
    
    jets.append(jet)
    
    Rotmom = Glob_mom_cons(jets)
    
    return e, m

inputfile = 'eejj_ECM206.lhe.gz'

events, weights, multiweights = readlhefile(inputfile)

emissions = []
momenta = []
nbins = 30 # The number of bins
Nevolve = 10000 # The number of evolutions
# Go over and find the emissions a set amount of times
#e, m = Shower_Events([], t0, aSover)
Pa = Particle(1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
Pa.E = Q
# Make the events in a for that works with my program. Dumb I know
Events = []

for event in events:
    
    newevent = []
    for p in event:
        P = Particle(p[0], 0, 0, np.sqrt(1), p[2], p[3], p[4], p[6], p[5], 0)
        newevent.append(P)
    Events.append(newevent)
    pass


ps = Evolve(Events[0][0], t0, aSover )

rot = rotate_momenta(Events[0][0], ps)

print(check_mom_cons(rot))

sys.exit()

for i in tqdm(list(range(Nevolve))):
   ps = Evolve(Pa, t0, aSover)
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
#plt.ylabel(r'$P_{qq}(z)(1-z)$')
#plt.title(r'$q \rightarrow qg$ Splitting Function')

# y label and title for q -> gq
plt.ylabel(r'$P_{qg}(z)z$')
plt.title(r'$q \rightarrow gq$ Splitting Function')

plt.show()