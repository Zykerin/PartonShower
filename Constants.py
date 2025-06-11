from alphaS import * # ALphaS coupling constant



Qc = 0.935 # The cuttoff scale to match Herwig's
Q = 1000 # The Hard scale? - the scale attributed to the hard process 
#aS= 0.118 # The coupling constant that is constant
aS = alphaS(0.118, 91.1876)
# The QCD constants
Nc = 3
Cf = (Nc**2 -1)/ (2 * Nc) # The quark color factor
Ca = Nc
Tr = 1/2


