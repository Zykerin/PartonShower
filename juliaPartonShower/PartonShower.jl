include("Constants.jl")
include("SplittingFunctions.jl")
include("LHEWriter.jl")
include("Structures.jl")
include("Shower.jl")
using LHEF
using ProgressBars



inputFile::String = "eejj_ECM206_1E6.lhe.gz"
#inputFile::String = "eejj_ECM206.lhe.gz"
outputFile::String = "juliaColorStructure.lhe"
#outputFile::String = "juliaColorStructureSmall.lhe"

events = parse_lhe(inputFile)


myEvents = []

for ev in events
    newEvent = Event([], [])
    for p in ev.particles
        newP = Particle(p.id, p.status, 0, 1, (p.m), 0, p.px, p.py, p.pz, p.e, 0, [0, 0, 0, 0], p.color1, p.color2, 1, 0, 0, true, "", [])
        push!(newEvent.Jets, newP)
    end
    push!(myEvents, newEvent)

end

showeredEvents = []

for (i, ev) in tqdm(enumerate(myEvents))
    newEvent = showerEvent(ev, pTmin, aSover)
    for p in newEvent.AllParticles

            if isnan(p.px)
                print(string(i) * "\n")
            end

    end
    push!(showeredEvents, newEvent)
end


showeredEV = []
# Turn my format into one that is readable by the LHEWriter
for ev in showeredEvents
    sParts = []
    for p in ev.AllParticles
        push!(sParts, [p.id, p.status, p.px, p.py, p.pz, p.E, p.m, p.color, p.antiColor])
    end
    push!(showeredEV, sParts)
end

ECM = 206
sigma = 1.2
error = 0.1

writeLHE(outputFile, showeredEV, ECM^2, ECM, sigma, error)
