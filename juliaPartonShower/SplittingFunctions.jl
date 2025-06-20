include("Constants.jl")



# The g -> gg splitting function, overestimate, integral, and integral inverse
function Pgg(z)
    return 2 * Ca * ((1 - z * (1-z))^2) / (z * (1-z))
end
function Pgg_over(z)
    return Ca * (1/(1-z) + 1/z)
end
function Intgg(z, aSover)
    return -Ca * (aSover) * log(1/z - 1)
end
function InvInt_gg(z, aSover)
    return 1 / (1 + exp(-z/ (Ca * aSover)))
end


# The g -> qqbar splitting function, overestimate, integral, and integral inverse
function Pgq(z)
    return Tr *(1- 2 * z * (1-z))
end
function Pgq_over(z)
    return Tr
end
function Int_gq(z, aSover)
    return Tr * (aSover /(2 * pi)) * z
end
function InvInt_gq(z, aSover)
    return z * 2 * pi / (Tr * aSover)
end


# The q -> gq splitting function, overestimate, integral, and integral inverse
function Pqg(z)
    return Pqq(1-z)
end
function Pqg_over(z)
    return Pqq_over(z)
end
function Int_qg(z, aSover)
    return 2 * Cf * (aSover / (2 * pi)) * log(z)
end
function InvInt_qg(z, aSover)
    return exp(2 * pi *z / (2 * Cf * aSover))
end


# The q -> qg Splitting Function, overestimate, integral, and integral inverse
function Pqq(z)
    return Cf * (1 + z^2)/(1 - z)
end
function Pqq_over(z)
    return Cf * 2 / (1 - z)
end
function Int_qq(z, aSover)
    return -2 * Cf * (aSover / (2 * pi)) * log(1-z)
end
function InvInt_qq(z, aSover)
    return 1- exp(-z / (2 * Cf * aSover / (2 * pi)))
end
