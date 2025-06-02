import random as rand
from alphaS import *
import numpy as  np
import scipy.optimize 
import matplotlib.pyplot as plt
import tqdm
from dataclasses import dataclass

# A class for the emission info
@dataclass
class emissioninfo:
    z: float
    t: float
    Ptsq:float
    Vmsq: float
    Generated: bool
    ContinueEvolve: bool

# A class for particles
@dataclass
class Particle:
    typ: int
    z_at_em: float
    t_at_em: float
    Pt: float
    Vm: float
    ContinueEvolution: bool = False

@dataclass
class Jet:
    Particles: list[Particle]

t0 = 1 # The cuttoff scale
Q = 1000 # The Hard scale? - the scale attributed to the hard process 
aS= 0.118 # The coupling constant
aSover = aS # Use the fixed overestimate alphaS constant
nbins = 30 # The number of bins
Nevolve = 10000 # The number of evolutions
# The QCD constants
Nc = 3
Cf = (Nc**2 -1)/ (2 * Nc) # The quark color factor
Ca = Nc
Tr = 1/2

# The g - > gg splitting function, overestimate, integral, and integral's inverse
def Pgg(z):
    return  2 * Ca * ((1 - z * (1-z))**2) / (z * (1-z))
    #return   Ca * (z / (1.0 - z) + (1.0 - z) / z + z * (1.0 - z))
def Pgg_over(z):
    return Ca * (1/(1-z) + 1/z)
def tGamma_gg(z, aSover):
    return -Ca * (aSover) * np.log(1/z - 1)
def inversetGamma_gg(z, aSover):
    return 1 / (1 + np.exp(-z/ (Ca * aSover)))


# The g -> qqbar splitting function
def Pgq(z):
    return Tr *(1- 2 * z * (1-z))
# The g -> qqbar splitting function overestimate
def Pgq_over(z):
    return Tr
# The g -> qqbar splitting function integral
def tGamma_gq(z, aSover):
    return Tr * (aSover) * z
# The g -> qqbar splitting function integral inverse
def inversetGamma_gq(z, aSover):
    return 1 / (Tr * aSover) * z


# The q -> gq splitting function, overestimate, integral, and inverse integral
def Pqg(z):
    return Pqq(1-z)
def Pqg_over(z):
    return Pqq_over(1-z)
def tGamma_qg(z, aSover):
    return tGamma_qq(1-z, aSover)
def inversetGamma_qg(z, aSover):
    return inversetGamma_qq(1-z, aSover)


# Define the q -> qg splitting function
def Pqq (z):
    return Cf * (  (1 + z**2) / (1 -z) )
# Define the overestimate of the p -> pq splitting function
def Pqq_over (z):
    return Cf * ( 2 / (1 - z))
# Define the analytical solution to the overestimate of t * Gamma integrated over z.
# For this program, this can be seen as rho tilda...
def tGamma_qq (z, aSover):
    return -2 * Cf * aSover * np.log(1 - z)
# Define the inverse gamma function
def inversetGamma_qq (z, aSover):
    return 1 - np.exp(-z / ( 2 * Cf * aSover))


# Define the transverse momnentum sqaured
def transversemmsq (t, z):
    return z**2 * (1 - z) **2 * t

# Functuon to find the virtual mass squared.t
def virtualmass (t, z):
    return z * (1 - z ) * t
# Define the upper and lower bounds of z, not the overestimate scale versiom
def zbounds (t, t0):
    return 1- np.sqrt( t0/t), np.sqrt(t0/t)


# Define the E(t) or Emission scale function
def E(t, tmax, R, aSover, t0, tGamma):
    
    zup, zlow = zbounds(t, t0)
    
    r =  tGamma(zup, aSover) - tGamma(zlow, aSover)
    return np.log(t / tmax**2) -  (1 /r) * np.log(R)



# Define the function to determine the t value
def tEmission(tmax, t0, R, aSover, tGamma):
    prec = 1E-3 # Precision for the solution
    argsol = (tmax, R, aSover, t0, tGamma)

    ContinuedEvolve = True
    t = scipy.optimize.ridder(E, 3.99, tmax**2, args = argsol, xtol= prec)
    
    # If a root is not found, stop the evolution for this branch
    if abs(E(t, tmax, R, aSover, t0, tGamma)) > prec:
        ContinuedEvolve = False
    return t, ContinuedEvolve



# Function to determine the z emission
def zEmission (t, t0, aSover, tGamma, inversetGamma):
    
    Rp = rand.random()
    zup, zlow = zbounds(t, t0)
    
    return inversetGamma( tGamma(zlow, aSover) + Rp * (tGamma(zup, aSover) - tGamma(zlow, aSover)), aSover )



# Define the function to generate the emissions
def GenerateEmissions (tmax, t0, aSover, branch_type):

    # Generate the three randomn numbers to determine whether or no to keep the solution
    R1 = rand.random()
    R2 = rand.random()
    
    Ems= emissioninfo(0,0, 0, 0, True, True)
    
    # Get the t emission value
    match branch_type:
        case 1:
            Ems.t, Ems.ContinueEvolve = tEmission(tmax, t0, R1, aSover, tGamma_gg)
        case 2:
            Ems.t, Ems.ContinueEvolve = tEmission(tmax, t0, R1, aSover, tGamma_qq)
        case 3:
            Ems.t, Ems.ContinueEvolve = tEmission(tmax, t0, R1, aSover, tGamma_gq)
            
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
    # Get the transverse momentum squared and virtual mass squared
    Ems.Ptsq = transversemmsq(Ems.t, Ems.z)
    Ems.Vmsq = virtualmass(Ems.t, Ems.z)
    # Determine whether the transverse momentum is physical
    if Ems.Ptsq < 0:
        Ems.generated = False
    
    # Determine whether or no to accept the t value, and inturn accept the emission
    match branch_type:
        case 1:
            if R2 > Pgg(Ems.z) / Pgg_over(Ems.z):
                    Ems.generated = False
        case 2:
            if R2 > Pqq(Ems.z) / Pqq_over(Ems.z):
                    Ems.enerated = False
        case 3:
            if R2 > Pgq(Ems.z) / Pgq_over(Ems.z):
                    Ems.Generated = False
    
    # If any of the tests are true, then there is no emission and return these values
    if Ems.Generated == False:
        Ems.z = 1
        Ems.Ptsq = 0
        Ems.Vmsq = 0  
    
    return Ems

# The function to evolve a certain branch
def Evolve(Q, t0, aSover):
    emissions = []
    # Cases: 1: g -> gg, 2: q -> qg, 3: g -> qqbar
    branch_type = 3
    
    t0 = t0**2
    tscalcuttoff = 4
    # Set the initial emission data class values
    Emission = emissioninfo(1, Q**2, 0, 0, True, True)
    # Loop until the evolution variable reaches the cuttoff 
    while np.sqrt(Emission.t) * Emission.z > np.sqrt( tscalcuttoff):
        # Code to determine which branch is in effect. WIP
        '''if Pa.typ <= 6 and Pa.typ >= -6:
            branch_type = 2
        elif Pa.typ == 21:
            Ems = GenerateEmissions(np.sqrt(t) * z, t0, aSover, branch_type)
        else:
            print('ERROR: Invalid Particle type')'''
        # Get the values.
        Emission = GenerateEmissions(np.sqrt(Emission.t) * Emission.z, t0, aSover, branch_type)

        # Terminate the evolution if a requirement is met.
        # This then stops this branch's evolution
        if Emission.ContinueEvolve == False:
            return emissions
        
        # Return emssions if the current one is out of bounds
        if Emission.t < ( tscalcuttoff):
            return emissions
        # Append the emissions and then continue
        if Emission.z != 1:
            emissions.append( (np.sqrt(Emission.t), Emission.z, np.sqrt(Emission.Ptsq), np.sqrt(Emission.Vmsq)))
        
    return emissions
   
emissions = []

# Go over and find the emissions a set amount of times
for i in tqdm.tqdm(list(range(Nevolve))):
    emish = Evolve(Q, t0, aSover)
    emissions = emissions + emish

# Set the arrays to obtain the physicals
ts = []
zs = []
Pt = []
Vm = []

# Get all the physicals
for var in emissions:
    ts.append(var[0])
    zs.append(var[1])
    Pt.append(var[2])
    Vm.append(var[3])

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


plt.plot(X, Y, label='generated', lw= 0, marker='o', markersize=4, color = 'blue')
plt.plot(X, testP, label='analytical', color = 'red')
plt.minorticks_on()

plt.xlabel('z')
plt.ylabel('P(z)')
plt.legend(loc='upper left')
#plt.title('g -> qqbar')
plt.title('g -> gg splitting function')
#plt.title('q -> qg')

plt.show()