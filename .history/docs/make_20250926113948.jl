using Documenter
using TightBindingToolbox

makedocs(
    build       =   "build" ,
    sitename    =   "TightBindingToolbox.jl"    ,
    modules     =   [TightBindingToolbox, TightBindingToolbox.UCell, TightBindingToolbox.DesignUCell, TightBindingToolbox.ExpandUCell, TightBindingToolbox.Parameters, TightBindingToolbox.LatticeStruct, TightBindingToolbox.DesignLattice, TightBindingToolbox.BZone, TightBindingToolbox.Hams, TightBindingToolbox.TBModel, TightBindingToolbox.Chern, TightBindingToolbox.suscep, TightBindingToolbox.conduct]   ,
    pages = [
        "Introduction"              =>  "index.md",
        "Unit Cell"                 =>  "UnitCell.md",
        "Parameters"                =>  "Params.md",
        "Lattice"                   =>  "Lattice.md",
        "Brillouin Zone"            =>  "BZ.md",
        "Hamiltonian"               =>  "Hamiltonian.md",
        "Tight Binding Model"       =>  "Model.md",
        "BdG Model"                 =>  "BdGModel.md",
        "Chern Numbers"             =>  "Chern.md",
        "Magnetic susceptibility"   =>  "susceptibility.md"  ,
        "Electrical Conductivity"   =>  "conductivity.md",
        "Plotting"                  =>  "Plot.md"
    ]
)

deploydocs(
    repo = "github.com/Anjishnubose/TightBindingToolbox.jl.git",
    devbranch = "main"
)