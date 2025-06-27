include("alphaS.jl")


# The QCD constants
#global const Nc = 3 # The number of colors
#global const Cf = (Nc^2 -1)/ (2 * Nc) # The quark color factor
#global const Ca = Nc
#global const Tr = 1/2



global const t0 = 1
global const Q = 1000
global const Qc = 0.935
aSover::Float64 = AlphaQ(Qc) # The overestimate for the coupling constants

#global const nbins: = 30
#global const Nevolve = 10000

global const pTmin = 0.65471
global const pT2min::Float64 = pTmin^2


a = 0.3
b = 2.3
delta = 2.8
c = 0.3

function Qg(mq::Float64)
   return max((delta - a * mq)/b, c)

end

function mufunc(mass::Float64, ps::Float64)
   return max(mass, Qg(ps))
end
