<h1> A Parton Shower Written in Julia</h1>

This is the parton shower in ``` Julia ``` written for research with Dr. Andreas Papaefstathiou. Our testing shower written in Python is also provided in this repository. However, it is not complete and does not provided and accurate shower. 


<h2>Pre-requisites for the Julia parton shower:</h2>
<ol>
<li> Julia 1.11</li>
<li> StatsBase, Roots, Random, ProgressBars, ArgParse, LHEF</li>
</ol>
<h2>Usage:</h2>

There are two parton showers here, one in Python and one in Julia. The one currently in usage is the one in Julia and the Python shower is not up to par. 

The ```Julia``` code showers $ e^+ e^- \rightarrow q\bar{q} $ events which can either be given a Les Houches Event (LHE) file or generate the hard proccess here. The program outputs to a LHE file.

<h3>Option 1: Read an LHE file</h3>

```
    julia PartonShower.jl [--lhefile] --infile [LHE file] outfile
```
<h3>Option 2: Generate events</h3>

``` 
    julia PartonShower.jl [--generate] -N [Number of events] outfile
```

Two example LHE files (```eejj_ECM206.lhe.gz``` and ```eejj_ECM206_1E6.lhe.gz```) can be found in the "dataFiles" directory. The first one has 10,000 events while the second has a larger 1,000,000 events. The ` .in ` files four our comparisons with ``` Herwig 7 ``` can also be found the then "dataFiles" directory.