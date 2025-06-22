import sys
from Constants import *
import scipy.optimize 



# Define the transverse momnentum sqaured.
# This includes options for the new and old ones.
def transversemmsq (t, z, branchType, mu):
    # Select which evolution variable to use.
    match EvolveType:
        case 'Old':
    
            return z**2 * (1 - z)**2 * t
        case 'QTilde':
            # Option for gluon radiating. Either g -> gg or g -> qqbar
            if branchType == 1 or branchType ==3:
                
                return z**2* (1-z)**2 * t**2 - mu[0]**2
                
            elif branchType ==2:
            
                return (1-z)**2 * (z**2 * t**2 - mu[0]**2) - z * mu[1]**2
            
            else:
                raise Exception('Invalid branch type')
        case _:
            raise Exception('Invalid evolution scale type')

# Functuon to find the virtual mass squared.t
def virtualmass (t, z):
    return z * (1 - z ) * t

# Define the upper and lower bounds of z, not the overestimate scale version
# Also include an option for the old evolution scale
def zBounds (masses, t, t0, branchType):
    # Select which evolution variable to use.
    if EvolveType == 'Old':
        return 1- np.sqrt( t0**2/t), np.sqrt(t0**2/t)
    elif EvolveType  == 'QTilde':
        mu = masses[0]
        Q = masses[1]
        # Option for gluon radiating. Either g -> gg or g -> qqbar
        if branchType == 1 or branchType ==3:
            #return 0.5 * (1 + np.sqrt(1 - 4 * np.sqrt( (mu**2 + pT2min)/t ))), 0.5 * (1 - np.sqrt(1 - 4 * np.sqrt((mu**2 + pT2min)/t)))
            return 1-np.sqrt((mu**2+pT2min))/t, np.sqrt(mu**2 + pT2min) / t
        # Option for quark radiating. For this program currently only q -> qg.
        elif branchType == 2:
            return  1- np.sqrt(Q**2 + pT2min)/t, np.sqrt(mu**2 + pT2min)/t
        else:
            raise Exception ('Invalid Branch type')
        
    else:
        raise Exception('Invalid evolution scale type')
    



# Function to determine the starting evolution scale
def EvolutionScale(p1, p2):
    
    match EvolveType :
        case 'Old':
            return [p1.E**2, p2.E**2]
        case 'QTilde':
            Q2 = ( p1.E + p2.E)**2 - (p1.Px + p2.Px)**2 - (p1.Py + p2.Py)**2 - (p1.Pz + p2.Pz)**2
            
            b = p1.m**2 / Q2
            c = p2.m**2/Q2
            lam = np.sqrt(1 + b**2 + c**2 - 2*b - 2*c - 2*b*c)
            ktildb = 0.5 * (1 + b - c + lam)
            ktildc = 0.5 * (1 - b + c + lam)
            
            Qtilde = [ np.sqrt(Q2 * ktildb), np.sqrt(Q2 * ktildc)]
            return [np.sqrt(Q2 * ktildb), np.sqrt(Q2 * ktildc)]



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
    
    zup, zlow = zBounds(mu, t, Qcut, branchType)
    r =  tGamma(zup, aSover) - tGamma(zlow, aSover)
    return np.log(t / Q**2) -  (1 /r) * np.log(Rp)



# Define the function to determine the t value
# This is doen by numerically solving the emission scale function
def tEmission(Q, Qcut, R2, aSover, tfac, tGamma, mu, branchType):
    prec = 1E-4 # Precision for the solution
    argsol = (Q, R2, aSover, Qcut, tGamma, mu,  branchType)
    
    ContinuedEvolve = True
    
    '''
    if branchType == 3 or branchType == 1:
        if Q < 16 * (mu[0]**2 + pT2min):
            ContinuedEvolve = False
            Q = -1
            return Q, ContinuedEvolve
    
    if zlow > zup:
        ContinuedEvolve = False
        Q = -1
        return Q, ContinuedEvolve
    '''
    #zup, zlow = zBounds(mu, Q, Qcut, branchType)

    if EvolveType == 'QTilde':
        #rho = tGamma(zup, aSover) - tGamma(zlow, aSover)
        #t = Q**2* R2**(1/rho)
        t = scipy.optimize.ridder(E, 4 * pT2min, Q**2, args = argsol, xtol= prec)
    else:
        t = scipy.optimize.ridder(E, tfac * Qcut**2 , Q**2, args = argsol, xtol= prec)
    
    # If a root is not found, stop the evolution for this branch
    if abs(E(t, Q, R2, aSover, Qcut, tGamma, mu, branchType)) > prec:
        t = Q**2
        ContinuedEvolve = False
    return t, ContinuedEvolve


# Function to determine the z emission
def zEmission (t, t0, Rp, aSover, tGamma, inversetGamma, mu, branchType):  

    zup, zlow = zBounds(mu, t, t0, branchType)
    
    z = inversetGamma( tGamma(zlow, aSover) + Rp * (tGamma(zup, aSover) - tGamma(zlow, aSover)), aSover )
    return z