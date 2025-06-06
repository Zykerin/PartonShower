import numpy as np
from dataclasses import dataclass, replace
import copy
import scipy.optimize 
import math

t0 = 1 # The cuttoff scale
Q = 1000 # The Hard scale? - the scale attributed to the hard process 
aS= 0.118 # The coupling constant
aSover = aS # Use the fixed overestimate alphaS constant
# The QCD constants
Nc = 3
Cf = (Nc**2 -1)/ (2 * Nc) # The quark color factor
Ca = Nc
Tr = 1/2



# Define the transverse momnentum sqaured
def transversemmsq (t, z):
    return z**2 * (1 - z)**2 * t

# Functuon to find the virtual mass squared.t
def virtualmass (t, z):
    return z * (1 - z ) * t
# Define the upper and lower bounds of z, not the overestimate scale versiom
def zbounds (t, t0):
    return 1- np.sqrt( t0/t), np.sqrt(t0/t)

# Get the rotation matrix
def rotationMatrix(v1, v2):
    k = np.cross(v1, v2)
    # Get the identity matrix
    I = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]]  )
    # Test if the norm of the cross product is zero
    if np.linalg.norm(k) == 0:
        return I
    else:
        k = k / np.linalg.norm(k)
        # For the K matrix
        K = np.array([[0, -k[2], k[1]], 
                      [k[2], 0, -k[0]],
                      [-k[1], k[0], 0]])
        # Get the angle between the two vectors
        theta = np.arccos(np.clip(np.dot(v1/np.linalg.norm(v1), v2/ np.linalg.norm(v2)), -1, 1))
        
        # Get the rotation matrix
        RotMat = I + np.sin(theta) * K + (1 - np.cos(theta)) *np.dot(K, K)
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

# Rotate a given particle with a given rotation matrix
def rotate(p, rotMat):
    
    pvec = [p.Px, p.Py, p.Pz]
    rotvec = np.dot(rotMat, pvec)
    
    rotp = copy.deepcopy(p)
    rotp.Px = rotvec[0]
    rotp.Py = rotvec[1]
    rotp.Pz = rotvec[2]
    return  rotp

# Rotate the given particles to allign with the mother particle's momentum
def rotate_momenta(p, particles):
    
    rotated_particles = []
    
    pmag = np.sqrt(p.Px**2 + p.Py**2 + p.Pz**2)
    Matrix = rotationMatrix([0, 0, pmag], [p.Px, p.Py, p.Pz])
    for p in particles:
        
        i = copy.deepcopy(p)
        v = np.array([i.Px, i.Py, i.Pz])
        #rotedVec = [(Matrix[0][0] + Matrix[0][1] + Matrix[0][1] )* v[0], (Matrix[1][0] + Matrix[1][1] + Matrix[1][2]) * v[1], (Matrix[2][0] + Matrix[2][1] + Matrix[2][2]) * v[2]]
        rotedVec = np.dot(Matrix, v)
        i.Px = rotedVec[0]
        i.Py = rotedVec[1]
        i.Pz = rotedVec[2]
        rotated_particles.append(i)
    return rotated_particles

# Get the boost factor beta for the outgouting jet (new) and parent jet (old)
def Boost_factor(k, new, old):
    #print(new, old)
    qs = new[0]**2 + new[1]**2 + new[2]**2
    q = np.sqrt(qs)
    Q2 = new[3]**2 - new[0]**2 - new[1]**2 - new[2]**2
    kp = k* np.sqrt(old[0]**2 + old[1]**2 + old[2]**2)
    kps= kp**2
    betamag = (q * new[3] - kp * np.sqrt(kps + Q2)) / (kps + qs + Q2)
    beta = betamag * (k / kp) * np.array([old[0], old[1], old[2]])
    #print(betamag, new)
    if betamag >= 0:
        return beta
    else:
        return [0, 0, 0]    

# The function to boost given particles with a given boost factor beta
def boost(particles, beta):
    
    bmag = np.sqrt(beta[0]**2 + beta[1]**2 + beta[2]**2)
    # Get the gamma factor
    gamma = 1 / np.sqrt(1 - bmag**2)
    #print(beta)
    boosted_particles = []
    # Go throught the entire list of particles and boost each one
    for p in particles:
    
        boosted_particle = copy.deepcopy(p)
    
        # Use the matrix of a lorentz boost from https://www.physicsforums.com/threads/general-matrix-representation-of-lorentz-boost.695941/
        boosted_particle.Px = (-gamma * beta[0] + (1 + (gamma - 1) * beta[0]**2 / bmag**2) + (gamma - 1) * beta[0] * beta[1] / bmag**2 + (gamma - 1) * beta[0] * beta[2] / bmag**2) * p.Px
        boosted_particle.Py = (- gamma * beta[1]  + (gamma - 1) * beta[1] * beta[0]/ bmag**2 + 1 + (gamma -1) * beta[1]**2 / bmag**2 + (gamma -1) * beta[1] * beta[2] / bmag**2) * p.Py
        boosted_particle.Pz = (-gamma * beta[2] + (gamma - 1) * beta[2] * beta[0] /bmag**2 + (gamma -1) * beta[2] * beta[1] / bmag**2 + 1 + (gamma -1) * beta[2]**2 / bmag**2) * p.Pz
        boosted_particle.E = (gamma - gamma * beta[0] - gamma * beta[1] - gamma * beta[2]) * p.E
         
        boosted_particles.append(boosted_particle)
        
    return boosted_particles

# The equation to numerically solve to find k
def K_eq(k, p, q, s):
    sump = 0
    # Loop through the momentum and sum up each term
    for i in range(len(p)):
        sump = sump + np.sqrt(k*k *p[i] + q[i])
        
    sump = sump - s
    
    return sump

# Solve for the k factor
def Solve_k_factor(pj, qj, s):
    
    kargs = (pj, qj, s)
    k = scipy.optimize.root(K_eq, 1.01, args = kargs)
    
    return k.x[0]

# Define the function to perfom global momentum conservation on the jets and particles
def Glob_mom_cons(Jets):
    
    #if len(Jets[0].Particles) == 0 :
    #    return [Particle(1, 0, 0, 0, 0, 0, 0, 0, 0, 0)]
    
    pj = [] # The momenta of the parent parton
    qj = [] # The momenta of the jets?
    newqs = []
    oldps = []
    rotms = [] # The rotatio matrices
    # Initialize the total energy
    sqrts = 0
    

    # Iterate through the list of jets
    for jet in Jets:
        
        #if len(jet.Particles)==0:
        #    continue
        
        # Append the 3-momentum of the Jet's progenitor to the 
        pj.append(jet.Progenitor.Px**2 + jet.Progenitor.Py**2 + jet.Progenitor.Pz**2)
        
        # Get the jet progenitor's 4 momentum
        oldp = np.array([jet.Progenitor.Px, jet.Progenitor.Py, jet.Progenitor.Pz, jet.Progenitor.E])
          
        # Add this jet's progenitor's energy
        sqrts += jet.Progenitor.E
        
        newq = [0, 0, 0, 0]
        # Iterate through the particles in the jet and add their momentum
        for p in jet.Particles:
            newp = copy.deepcopy(p)
            newq[0] = newq[0] + newp.Px
            newq[1] = newq[1] + newp.Py
            newq[2] = newq[2] + newp.Pz
            newq[3] = newq[3] + newp.E
        
        # Calculate the momentum and append it to the list of jet momentum
        qj2 = newq[3]**2 - newq[0]**2 - newq[1]**2 - newq[2]**2
        if math.isnan(qj2):
            qj2 = 0
        Rqp = rotationMatrix(np.array([newq[0], newq[1], newq[2]]), np.array([oldp[0], oldp[1], oldp[2]] ))
        oldps.append(oldp)
        
        newqs.append(newq)
        qj.append(qj2)
        rotms.append(Rqp)
        
    
    # Get the k factor
    k = Solve_k_factor(pj, qj, sqrts)
    boosted_particles = []
    # Iterate though the jets list and boost each
    for i, jet in enumerate(Jets):
        
        if len(jet.Particles) == 0:
            boosted_particles.append(jet.Progenitor)
        else:
            
            jet = copy.deepcopy(jet)
           # if len(jet.Particles)==0:
            #    continue
        
            for p in jet.Particles:
                #return rotated
                rotated = rotate(p, rotms[i])
                # Get boosted
                beta = Boost_factor(k, newqs[i], oldps[i])
                boosted_particles.append(boost([rotated], beta))
            
    return boosted_particles