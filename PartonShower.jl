using PyPlot

Cf = 4/3
Ca = 3
Tr = 1/2
t0 = 1
Q = 1000
aS = 0.118
aSover = aS
nbins = 30
Nevolve = 10000


function Pgg(z)
    return 2 * Ca * ((1 - z * (1-z))^2) / (z * (1-z))
end

function Pgg_over(z)
    return Ca * (1/(1-z) + 1/z)
end

function tGamma_gg(z, aSover)
    return -Ca * (aSover) * log(1/z - 1)
end

function inversetGamma_gg(z, aSover)
    return 1 / (1 + exp(-z/ (Ca * aSover)))
end
