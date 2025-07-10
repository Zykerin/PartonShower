include("Constants.jl")
include("SplittingFunctions.jl")
include("LHEWriter.jl")
include("Structures.jl")
include("Shower.jl")
include("HardProccessEventGenerator.jl")
using LHEF
using ProgressBars

# Variable to decide whether or not to generate the events from the hard proccess as well or read from an already generated lhe file 
# Options are: "generate" or "lhe"
whichEvents::String = "generate"

outputFile::String = "juliaColorStructure.lhe"
#outputFile::String = "juliaColorStructureSmall.lhe"
error = 0.1
sigma = 1.2
myEvents = []

if whichEvents == "lhe"
    print("LHE file selected. \n")
    error = 0.1
    sigma = 1.2
    inputFile::String = "eejj_ECM206_1E6.lhe.gz"
    #inputFile::String = "eejj_ECM206.lhe.gz"
    
    print("Reading Events \n")
    events = parse_lhe(inputFile)

    for ev in events
        newEvent = Event([], [])
        for p in ev.particles
            newP = Particle(p.id, p.status, 0, 1, (p.m), 0, p.px, p.py, p.pz, p.e, 0, [0, 0, 0, 0], p.color1, p.color2, 1, 0, 0, true, "", [])
            push!(newEvent.Jets, newP)
        end
        push!(myEvents, newEvent)

    end
elseif whichEvents == "generate"
    print("Generate events selected. \n")
    print("Generating events\n")
    myEvents, sigma, error = generateEvents(1E4, 1E6)
end
showeredEvents = []


print("Showering events \n")
for (i, ev) in tqdm(enumerate(myEvents))
    newEvent = showerEvent(ev, pTmin, aSover)
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



writeLHE(outputFile, showeredEV, ECM^2, ECM, sigma, error)
