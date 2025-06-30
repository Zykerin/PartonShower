include("Constants.jl")
include("SplittingFunctions.jl")
include("LHEWriter.jl")
include("Structures.jl")
include("Shower.jl")
using PyCall
using LHEF
using ProgressMeter

@pyimport sys 


#pushfirst!(PyVector(pyimport("sys")."path"), "")
#@pyimport LHEReader


inputFile::String = "eejj_ECM206.lhe.gz"
outputFile::String = "Tester.lhe"

events = parse_lhe(inputFile)


myEvents = []

for ev in events
    newEvent = Event([], [])
    for p in ev.particles
        newP = Particle(p.id, p.status, 0, 1, p.m, 0, p.px, p.py, p.pz, p.e, 0, [p.px, p.py, p.pz, p.e], p.color1, p.color2, 1, 0, true, "", [])
        push!(newEvent.Jets, newP)
    end
    push!(myEvents, newEvent)

end

showeredEvents = []

@showprogress for ev in myEvents
    newEvent = showerEvent(ev, pTmin, aSover)
    push!(showeredEvents, newEvent)
end

showeredEV = []
# Turn my format into one that is readable by the LHEWriter
for ev in showeredEvents
    sParts = []
    for p in ev.AllParticles
        push!(sParts, [p.id, p.status, p.px, p.py, p.pz, p.E, p.m])
    end
    push!(showeredEV, sParts)
end

ECM = 206
sigma = 1.2
error = 0.1

writeLHE(outputFile, showeredEV, ECM^2, ECM, sigma, error)
