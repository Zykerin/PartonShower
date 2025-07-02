global const Nc = 3 # The number of colors
global const Cf = (Nc^2 -1)/ (2 * Nc) # The quark color factor associated with gluon emissions from a quark
global const Ca= Nc # The color factor associated with gluon emissions from another gluon
global const Tr = 1/2 # The color facor for a gluon to qqbar emission

#asmz = 0.118
asmz = 0.1074
mz = 91.1876
order = 1
mb = 4.75
mc = 1.27
order = 1
mz2 = mz^2
mc2 = mc^2
mb2 = mb^2

#=
function GetAlphaS(t::Float64, z::Float64)
    mb = 4.75
    mc = 1.27
    order = 1
    mz2 = mz^2
    mc2 = mc^2
    mb2 = mb^2
    asmb = AlphaQ(mb)
    asmc = AlphaQ(mc)
    scale = z * (1 -z) * sqrt(t)
    if scale < Qc
        scale = Qc
    end
    Q = scale^2
    if Q>= mb2
        tref = mz2
        asref = asmz
        b0 = Beta0(5) / (2 * pi)
        b1 = Beta1(5)/ (2 * pi)^2
    elseif Q >= mc2
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
=#



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

asmb = AlphaQ(mb)
asmc = AlphaQ(mc)


function getAlphaS(t::Float64, z::Float64, Qcut::Float64)
    scale = z * (1- z) * sqrt(t)
    if scale < Qcut
        scale = Qcut
    end
    return AlphaQ(scale)
    
end

