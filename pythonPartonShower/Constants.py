from alphaS import * # ALphaS coupling constant


#Qc = 1 # Old cutoff scale for testing
Qc = 0.935 # The cuttoff scale to match Herwig's
Q = 1000 # The Hard scale? - the scale attributed to the hard process 
#aS= 0.118 # The coupling constant that is constant
aS = alphaS(0.118, 91.1876)

# The QCD constants
Nc = 3 # The numer of colors in QCD
Cf = (Nc**2 -1)/ (2 * Nc) # The quark color factor
Ca = Nc
Tr = 1/2

GeV = 1

# Masses of the quarks
m1 = 4.8 # Down quark
m2 = 2.3 # Up quark
m3 = 95 # Strange quark
m4 = 1275 # Charm quark
m5 = 4180 # Bottom quark
m6 = 173210 # Top quark


# Parameters for gluon's minimum virtuality.
# These are obtained from Herwig
a = 0.3
b = 2.3
delta = 2.8 # Aka cutoffKinScale
c = 0.3

debug = False

pTmin = 0.65471
pT2min = pTmin**2

# Select the evolution variable, either 'QTilde' for Herwig's evolution variable, or 'Old' for the one this program was using before i.e E^2
EvolveType = 'QTilde' 


# Define the function to return the gluon's minimum virtuality
# q is the quark that is radiating
def Qg(mq):

    Q = max((delta - a * mq)/ b, c)  
    return Q

# Define the function to return the mu value for a given parent particle's mass and a given quark.
def mufunc(mass, ps):
    return max(mass, Qg(ps)) 






