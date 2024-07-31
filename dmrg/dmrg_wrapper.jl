
"""
This code performs an example DMRG run to compute the binder cumulant or
fidelity, sweeping through a range of transverse fields and saving the output
in the .jls format
"""

using ITensors
using Serialization

include("dmrg_functions.jl")

"""
Setting parameters for the calculation (see dmrg_functions.jl for a description
of paraemters)
"""
N = 100
interaction_sign = "FM"
lamX = 0.1
lamY = 0.0
lamZ = 0.0
lamXX = 0.0
lamYY = 0.0
lamZZ = 0.0
lamAD = 0.0
nsweeps = 30
maxdim = [10,20,100,100,200]
cutoff = [1E-10]
psi0_bonddim = 30

"a run computing the binder cumulant for a range of transverse fields"
data = [(
    round(g, digits=4),
    dmrg_binder(
        N,
        interaction_sign,
        g,
        lamX,
        lamY,
        lamZ,
        lamXX,
        lamYY,
        lamZZ,
        lamAD,
        nsweeps,
        maxdim,
        cutoff,
        psi0_bonddim
    )
) for g in LinRange(0.025,1.25,50)] # sweep over a range of transverse fields
serialize(
    "data/dmrg_binder_N$(N)_$(interaction_sign)_lamX$(lamX)_lamY$(lamY)_"
    "lamZ$(lamZ)_lamXX$(lamXX)_lamYY$(lamYY)_lamZZ$(lamZZ)_lamAD$(lamAD)_"
    "nsweeps$(nsweeps)_psi0BD$(psi0_bonddim).jls", data
)    #saving data


"a run computing the fidelity for a range of transverse fields"

# """
# setting up the value of the symmetry-breaking field at the ends of the spin
# ladder (see dmrg_functions.jl for a description of paraemters)
# """
# lamSBends=0.1
#
data = [(
    round(g, digits=4),
    dmrg_fidelity(
        N,
        interaction_sign,
        g,
        lamX,
        lamY,
        lamZ,
        lamXX,
        lamYY,
        lamZZ,
        lamAD,
        lamSBends,
        nsweeps,
        maxdim,
        cutoff,
        psi0_bonddim
    )
) for g in LinRange(0.025,1.25,50)]  # sweep over a range of transverse fields
serialize(
    "data/dmrg_fidelity_N$(N)_$(interaction_sign)_lamX$(lamX)_lamY$(lamY)_"
    "lamZ$(lamZ)_lamXX$(lamXX)_lamYY$(lamYY)_lamZZ$(lamZZ)_lamAD$(lamAD)_"
    "lamSBends$(lamSBends)_nsweeps$(nsweeps)_psi0BD$(psi0_bonddim).jls", data
)    #saving data
