import numpy as np

t0 = 0.935 # The cuttoff scale
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
    #return  Ca * ((1 - z * (1-z))**2) / (z * (1-z))
    return   Ca * (z / (1.0 - z) + (1.0 - z) / z + z * (1.0 - z))
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
