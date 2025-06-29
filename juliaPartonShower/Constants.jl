include("alphaS.jl")






#global const t0 = 1
#global const Q = 1000
global const Qc = 0.935
#global const Qc::Float64 = 0.65471
aSover::Float64 = AlphaQ(Qc) # The overestimate for the coupling constants

global const pTmin::Float64 = 0.65471
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
