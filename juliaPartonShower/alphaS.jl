global const Nc = 3 # The number of colors
global const Cf = (Nc^2 -1)/ (2 * Nc) # The quark color factor
global const Ca = Nc
global const Tr = 1/2

#=
Base.@kwdef mutable struct alphaS
    asmz::Float64
    mz::Float64
    mb::Float64 = 4.75
    mc::Float64 = 1.27
    order::Int = 1
    mz2::Float64 = mz^2>
    mc2::Float64 = mc^2
    mb2::Float64 = mb^2
    #alphaS() = new()
    #asmb::Float64 = AlphaQ(alpha, mb)
    #asmc::Float64 = AlphaQ(self, mc)
end
=#
asmz = 0.118
mz = 91.1876
mb = 4.75
mc = 1.27
order = 1
mz2 = mz^2
mc2 = mc^2
mb2 = mb^2


#struct alphaS
#    asmz::Float64
#    mz::Float64
#end


function As1(t::Float64)
    if t>= mb2
        tref = mz2
        asref = asmz
        b0 = Beta0(5) / (2 * pi)
        b1 = Beta1(5)/ (2 * pi)^2
    elseif t >= mc2
        tref = mb2
        asref = asmb
        b0 = Beta0(4) / (2 * pi)
        b1 = Beta1(4)/ (2 * pi)^2
    else
        tref = mc2
        asref = asmc
        b0 = Beta0(3) / (2 * pi)
        b1 = Beta1(3)/ (2 * pi)^2
    end
    w = 1 + b0 * asref * log(t/tref)
    return asref / w * (1 - b1 / b0 * asref * log(w) / w)
end


function Beta0(nf::Int)
    return (11/ 6 * Ca) - (2/3 * Tr * nf)
end
function Beta1(nf::Int)
    return (17 / 6 * Ca^2) - ( (5 / 3 * Ca + Cf) * Tr * nf )
end


function AlphaQ(Q::Float64)
   if order == 0
       return As1(Q^2)
   else
       return As1(Q^2)
   end
end


function GetAlphaS(t, z)
   scale = z * (1 -z) * sqrt(t)
   if scale < Qc
       return AlphaQ(Qc)
   else
       return AlphaQ(scale)
   end
end

asmb = AlphaQ(mb)
asmc = AlphaQ(mc)


