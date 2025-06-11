from alphaS import * # ALphaS coupling constant
import numpy as  np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys 
from LHEWriter import * # part to write to a lhe file
from LHEReader import * # part to read the lhe file
from SplittingFunctions import * # Splitting functions file
from Kinematics import * # Kinematics reconstuction file
from Classes import * # File containing data classes
from Constants import * # File containing the constants
from Shower import * # File containing the showering functions

aSover = GetalphaSOver(Qc)
#aSover = 0.118
# Get the input file, read and then parse it
inputfile = 'eejj_ECM206.lhe.gz'

events, weights, multiweights = readlhefile(inputfile)

emissions = [] 
nbins = 30 # The number of bins
Nevolve = 100000 # The number of evolutions

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

'''
ShoweredEvents = []
for event in tqdm(Events):
    
    Ev = ShowerEvent(event, Qc, aSover)
    ShoweredEvents.append(Ev)

pss = []
for ev in tqdm(Events):
    ps = Shower_Evens(ev, Qc, aSover)
    pss.append(ps)
'''

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

Y = np.array(dist) * (1 - X)
testP = Pqq(X) * (1 - X)

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