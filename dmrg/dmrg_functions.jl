
"""
This code computes the ground state and the gap of an Ising spin ladder
Hamiltonian using DMRG.
The DMRG-computed ground state is used to obtain the Binder cumulant and the
fidelity of the mixed steady state of the noisy ITE process in the limit of
infinitesimally small Trotter step.

Reference: https://arxiv.org/abs/2406.04285.

Packages information:
---------------------
ITensors version = 0.3.57
Julia version = 1.10.1
"""


function dmrg_binder(
    N :: Int,
    interaction_sign :: String,
    g :: Float64,
    lamX :: Float64,
    lamY :: Float64,
    lamZ :: Float64,
    lamXX :: Float64,
    lamYY :: Float64,
    lamZZ :: Float64,
    lamAD :: Float64,
    nsweeps :: Int,
    maxdim,
    cutoff,
    psi0_bonddim :: Int
)
    """
    Computes ground state of an Ising spin ladder Hamiltonian using DMRG and
    the corresponding Binder cumulant of the mixed steady state of the noisy
    ITE process.

    Parameters
    ----------
    N : Int
        The total number of sites on the ladder.
    interaction_sign : String
        Interaction along the legs of the ladder.
        Relevant options: FM or AFM
    g : Float64
        Transverse field.
    lamX : Float64
        XX-type interleg coupling (induced by X noise in the ITE).
    lamY : Float64
        YY-type interleg coupling (induced by Y noise in the ITE).
    lamZ : Float64
        ZZ-type interleg coupling (induced by Z noise in the ITE).
    lamXX : Float64
        XXXX-type plaquette interleg coupling (induced by XX noise in the ITE).
    lamYY : Float64
        YYYY-type plaquette interleg coupling (induced by YY noise in the ITE).
    lamZZ : Float64
        ZZZZ-type plaquette interleg coupling (induced by ZZ noise in the ITE).
    lamAD : Float64
        AD noise induced interleg coupling.
    nsweeps : Int
        The number of DMRG sweeps.
    maxdim :
        The maximum DMRG bod dimension.
    cutoff :
        DMRG cutoff.
    psi0_bonddim : Int
        Bond dimension of a random initial MPS in DMRG.
    """
    sites = siteinds("S=1/2",N)

    "Setting a magnetization on the upper ladder (squared) MPO."
    os_mag2 = OpSum()
    for i=1:2:N-1, j=1:2:N-1
        if interaction_sign=="FM"
            os_mag2 += "Z",i,"Z",j
        end
        if interaction_sign=="AFM"
            os_mag2 += (-1)^((i+1)/2+(j+1)/2),"Z",i,"Z",j
        end
    end
    mag2 = MPO(os_mag2, sites)

    "Setting a spin ladder Hamiltonian MPO."
    os = OpSum()
    for j=1:2:N-2   # upper leg of the ladder
        if interaction_sign=="FM"
            os += -1,"Z",j,"Z",j+2
        end
        if interaction_sign=="AFM"
            os += +1,"Z",j,"Z",j+2
        end
        os += g,"X",j
    end
    os += g,"X",N-1
    for j=2:2:N-2   # lower leg of the ladder
        if interaction_sign=="FM"
            os += -1,"Z",j,"Z",j+2
        end
        if interaction_sign=="AFM"
            os += +1,"Z",j,"Z",j+2
        end
        os += g,"X",j
    end
    os += g,"X",N
    for j=1:2:N-1   # interleg coupling
        os += -lamX,"X",j,"X",j+1
        os += lamY,"Y",j,"Y",j+1
        os += -lamZ,"Z",j,"Z",j+1
    end
    for j=1:2:N-3   # interleg coupling
        os += -lamXX,"X",j,"X",j+1,"X",j+2,"X",j+3
        os += -lamYY,"Y",j,"Y",j+1,"Y",j+2,"Y",j+3
        os += -lamZZ,"Z",j,"Z",j+1,"Z",j+2,"Z",j+3
    end
    for j=1:N   # AD-noise induced term
        os += -lamAD,"Z",j
    end
    for j=1:2:N-1   # AD-noise induced terms
        os += -lamAD,"X",j,"X",j+1
        os += lamAD,"Y",j,"Y",j+1
        os += -im*lamAD,"X",j,"Y",j+1
        os += -im*lamAD,"Y",j,"X",j+1
    end
    H = MPO(os, sites)

    "Setting a sum_i Z_iZ_{i+1} MPO along the rungs of the ladder."
    os_obs_t = OpSum()
    for j=1:2:N-1
        os_obs_t += "Z",j,"Z",j+1
    end
    obs_t = MPO(os_obs_t, sites)

    """
    DMRG routine computing the ground state of the spin ladder.
    The routine is performed multiple times with a random initial state until
    the convergence criteria are met or the maximum number of iterations is
    reached.
    This is needed because sometimes DMRG can converge to a state within the
    ground state manifold that has a very small overlap with the |1>> state.
    The small overlap can lead to numerical errors.
    """
    for i in 1:100
        psi0 = randomMPS(sites, psi0_bonddim)
        if lamAD == 0.0
            energy_l, psi_l = dmrg(
                H,psi0;
                nsweeps,
                maxdim,
                cutoff,
                outputlevel=0,
                ishermitian=true
            )
        else
            "The Hamiltonian is non-Hermitian if the AD noise is present."
            energy_l, psi_l = dmrg(
                H,psi0;
                nsweeps,
                maxdim,
                cutoff,
                outputlevel=0,
                ishermitian=false
            )
        end
        global energy = energy_l
        global psi = psi_l
        """
        The first criterion ensures that the overlap between the ground state
        and the |1>> state is nonzero, i.e. that DMRG converged to the correct
        symmetry-broken state.
        The second criterion is added for the case of a disordered ground state.
        """
        if (real(inner(psi_l',obs_t,psi_l)) > 0 ||
            abs(real(inner(psi_l',obs_t,psi_l))) < 0.001)
            break
        end
    end

    "Setting an MPS for the |1>> state."
    A = [1 0
        0 1]
    id_efd_mps = MPS(sites,"0")
    for i in 1:div(N,2)
        Bell_MPS = MPS(A,sites[2*i-1:2*i],maxdim=2)
        id_efd_mps[2*i-1]=Bell_MPS[1]
        id_efd_mps[2*i]=Bell_MPS[2]
    end

    """
    Calculating the mixed-state expectation value of <m^4> and <m^2> using the
    doubled space formalism.
    """
    avg4 = inner(mag2,id_efd_mps,mag2,psi)/inner(id_efd_mps,psi)
    avg2 = inner(id_efd_mps',mag2,psi)/inner(id_efd_mps,psi)

    "Outputting the Binder cumulant and the ground state energy."
    return real(3/2 - 1/2*avg4/avg2^2), energy
end


function dmrg_fidelity(
    N :: Int,
    interaction_sign :: String,
    g :: Float64,
    lamX :: Float64,
    lamY :: Float64,
    lamZ :: Float64,
    lamXX :: Float64,
    lamYY :: Float64,
    lamZZ :: Float64,
    lamAD :: Float64,
    lamSBends :: Float64,
    nsweeps :: Int,
    maxdim,
    cutoff,
    psi0_bonddim :: Int
)

    """
    Computes ground state of an Ising spin ladder Hamiltonian with and
    without noise-induced couplings using DMRG, and evaluates the overlap
    beetween the two ground states.
    This overlap corresponds to the fidelity between the steady states of the
    noiseless and the noisy ITE.
    For the smaller system sizes that we've checked (N=50,100,200) and despite
    the symeetry breaking fields added at the ends of the ladder, the DMRG still
    sometimes picks ground states for H0 and H from different SB sectors, which
    results in a very small fidelity.
    In this case, the DMRG calculation is needed to be performed again with a
    different initial state. This issue seems to disappear for larger system
    sizes (N=400,800)

    Parameters
    ----------
    N : Int
        The total number of sites on the ladder.
    interaction_sign : String
        Interaction along the legs of the ladder.
        Relevant options: FM or AFM
    g : Float64
        Transverse field.
    lamX : Float64
        XX-type interleg coupling (induced by X noise in the ITE).
    lamY : Float64
        YY-type interleg coupling (induced by Y noise in the ITE).
    lamZ : Float64
        ZZ-type interleg coupling (induced by Z noise in the ITE).
    lamXX : Float64
        XXXX-type plaquette interleg coupling (induced by XX noise in the ITE).
    lamYY : Float64
        YYYY-type plaquette interleg coupling (induced by YY noise in the ITE).
    lamZZ : Float64
        ZZZZ-type plaquette interleg coupling (induced by ZZ noise in the ITE).
    lamAD : Float64
        AD noise induced interleg coupling.
    lamSBends : Float64
        Symmetry-breaking field added at the ends of the spin ladder to ensure
        that the DMRG picks a consistent summetry-broken grounds state.
    nsweeps : Int
        The number of DMRG sweeps.
    maxdim :
        The maximum DMRG bod dimension.
    cutoff :
        DMRG cutoff.
    psi0_bonddim : Int
        Bond dimension of a random initial MPS in DMRG.
    """

    sites = siteinds("S=1/2",N)

    "Setting a spin ladder Hamiltonian MPO."
    os = OpSum()
    for j=1:2:N-2   # upper leg of the ladder
        if interaction_sign=="FM"
            os += -1,"Z",j,"Z",j+2
        end
        if interaction_sign=="AFM"
            os += +1,"Z",j,"Z",j+2
        end
        os += g,"X",j
    end
    os += g,"X",N-1
    for j=2:2:N-2   # lower leg of the ladder
        if interaction_sign=="FM"
            os += -1,"Z",j,"Z",j+2
        end
        if interaction_sign=="AFM"
            os += +1,"Z",j,"Z",j+2
        end
        os += g,"X",j
    end
    os += g,"X",N

    os += -lamSBends,"Z",1   # adding symmetry-breaking field
    os += -lamSBends,"Z",2   # adding symmetry-breaking field
    os += -lamSBends,"Z",N-1   # adding symmetry-breaking field
    os += -lamSBends,"Z",N   # adding symmetry-breaking field

    "Creating MPO for the Hamiltonian without the noise-induced coupling."
    H0 = MPO(os,sites)

    for j=1:2:N-1   # interleg coupling
        os += -lamX,"X",j,"X",j+1
        os += lamY,"Y",j,"Y",j+1
        os += -lamZ,"Z",j,"Z",j+1
    end
    for j=1:2:N-3   # interleg coupling
        os += -lamXX,"X",j,"X",j+1,"X",j+2,"X",j+3
        os += -lamYY,"Y",j,"Y",j+1,"Y",j+2,"Y",j+3
        os += -lamZZ,"Z",j,"Z",j+1,"Z",j+2,"Z",j+3
    end
    for j=1:N   # AD-noise induced term
        os += -lamAD,"Z",j
    end
    for j=1:2:N-1   # AD-noise induced terms
        os += -lamAD,"X",j,"X",j+1
        os += lamAD,"Y",j,"Y",j+1
        os += -im*lamAD,"X",j,"Y",j+1
        os += -im*lamAD,"Y",j,"X",j+1
    end

    "Creating MPO for the Hamiltonian with the noise-induced coupling."
    H = MPO(os,sites)

    "Setting a sum_i Z_iZ_{i+1} MPO along the rungs of the ladder."
    os_obs_t = OpSum()
    for j=1:2:N-1
        os_obs_t += "Z",j,"Z",j+1
    end
    obs_t = MPO(os_obs_t,sites)

    """
    DMRG routine computing the ground state of H0.
    The routine is performed multiple times with a random initial state until
    the convergence criteria are met or the maximum number of iterations is
    reached.
    This is needed because sometimes DMRG can converge to a state within the
    ground state manifold that has a very small overlap with the |1>> state.
    The small overlap can lead to numerical errors.
    """
    for i in 1:100
        psi0 = randomMPS(sites,psi0_bonddim)
        if lamAD == 0.0
            energy_l, psi_l = dmrg(
                H0,psi0;
                nsweeps,
                maxdim,
                cutoff,
                outputlevel=0,
                ishermitian=true
            )
        else
            "The Hamiltonian is non-Hermitian if the AD noise is present."
            energy_l, psi_l = dmrg(
                H0,psi0;
                nsweeps,
                maxdim,
                cutoff,
                outputlevel=0,
                ishermitian=false
            )
        end
        global energy_gs = energy_l
        global psi_gs = psi_l
        """
        The first criterion ensures that the overlap between the ground
        state and the |1>> state is nonzero, i.e. that DMRG converged to the
        correct symmetry-broken state.
        The second criterion is added for the case of a disordered ground state.
        """
        if (real(inner(psi_l',obs_t,psi_l)) > 0 ||
            abs(real(inner(psi_l',obs_t,psi_l))) < 0.001)
            break
        end
    end

    """
    DMRG routine computing the ground state of H.
    The routine is performed multiple times with a random initial state until
    the convergence criteria are met or the maximum number of iterations is
    reached.
    This is needed because sometimes DMRG can converge to a state within the
    ground state manifold that has a very small overlap with the |1>> state.
    The small overlap can lead to numerical errors.
    """
    for i in 1:100
        psi0 = randomMPS(sites,psi0_bonddim)
        if lamAD == 0.0
            energy_l, psi_l = dmrg(
                H,psi0;
                nsweeps,
                maxdim,
                cutoff,
                outputlevel=0,
                ishermitian=true
            )
        else
            "The Hamiltonian is non-Hermitian if the AD noise is present."
            energy_l, psi_l = dmrg(
                H,psi0;
                nsweeps,
                maxdim,
                cutoff,
                outputlevel=0,
                ishermitian=false
            )
        end
        global energy = energy_l
        global psi = psi_l
        """
        The first criterion ensures that the overlap between the ground state
        and the |1>> state is nonzero, i.e. that DMRG converged to the correct
        symmetry-broken state.
        The second criterion is added for the case of a disordered ground state.
        """
        if (real(inner(psi_l',obs_t,psi_l)) > 0 ||
            abs(real(inner(psi_l',obs_t,psi_l))) < 0.001)
            break
        end
    end

    "Outputting the overlap between the two ground states."
    return abs(inner(psi_gs',psi))
end


function dmrg_gap(
    N :: Int,
    interaction_sign :: String,
    g :: Float64,
    lamX :: Float64,
    lamY :: Float64,
    lamZ :: Float64,
    lamXX :: Float64,
    lamYY :: Float64,
    lamZZ :: Float64,
    lamAD :: Float64,
    nsweeps :: Int,
    maxdim,
    cutoff,
    psi0_bonddim :: Int,
    weight :: Float64
)
    """
    Computes gap between the ground and the first excited state of an Ising spin
    ladder Hamiltonian using DMRG.

    Parameters
    ----------
    N : Int
        The total number of sites on the ladder.
    interaction_sign : String
        Interaction along the legs of the ladder.
        Relevant options: FM or AFM
    g : Float64
        Transverse field.
    lamX : Float64
        XX-type interleg coupling (induced by X noise in the ITE).
    lamY : Float64
        YY-type interleg coupling (induced by Y noise in the ITE).
    lamZ : Float64
        ZZ-type interleg coupling (induced by Z noise in the ITE).
    lamXX : Float64
        XXXX-type plaquette interleg coupling (induced by XX noise in the ITE).
    lamYY : Float64
        YYYY-type plaquette interleg coupling (induced by YY noise in the ITE).
    lamZZ : Float64
        ZZZZ-type plaquette interleg coupling (induced by ZZ noise in the ITE).
    lamAD : Float64
        AD noise induced interleg coupling.
    nsweeps : Int
        The number of DMRG sweeps.
    maxdim :
        The maximum DMRG bod dimension.
    cutoff :
        DMRG cutoff.
    psi0_bonddim : Int
        Bond dimension of a random initial MPS in DMRG.
    weight : Float64
        Weight to multiply the overlap between the states when minimizing it
        (see ITensor documentation).
    """
    sites = siteinds("S=1/2",N)

    "Setting a spin ladder Hamiltonian MPO."
    os = OpSum()
    for j=1:2:N-2    # upper leg of the ladder
        if interaction_sign=="FM"
            os += -1,"Z",j,"Z",j+2
        end
        if interaction_sign=="AFM"
            os += +1,"Z",j,"Z",j+2
        end
        os += g,"X",j
    end
    os += g,"X",N-1
    for j=2:2:N-2    # lower leg of the ladder
        if interaction_sign=="FM"
            os += -1,"Z",j,"Z",j+2
        end
        if interaction_sign=="AFM"
            os += +1,"Z",j,"Z",j+2
        end
        os += g,"X",j
    end
    os += g,"X",N
    for j=1:2:N-1    # interleg coupling
        os += -lamX,"X",j,"X",j+1
        os += lamY,"Y",j,"Y",j+1
        os += -lamZ,"Z",j,"Z",j+1
    end
    for j=1:2:N-3    # interleg coupling
        os += -lamXX,"X",j,"X",j+1,"X",j+2,"X",j+3
        os += -lamYY,"Y",j,"Y",j+1,"Y",j+2,"Y",j+3
        os += -lamZZ,"Z",j,"Z",j+1,"Z",j+2,"Z",j+3
    end
    for j=1:N    # AD-noise induced term
        os += -lamAD,"Z",j
    end
    for j=1:2:N-1    # AD-noise induced term
        os += -lamAD,"X",j,"X",j+1
        os += lamAD,"Y",j,"Y",j+1
        os += -im*lamAD,"X",j,"Y",j+1
        os += -im*lamAD,"Y",j,"X",j+1
    end
    H = MPO(os,sites)
    """
    In the symmetry-broken phase, the ground state is four-fold degenerate and
    one needs to find the first five lowest energy states to calcualate the gap.
    In the disordered phase, however, there is no ground state degeneracy.
    Below the number of eignestates to be computed depends on which phase the
    system is in; however, the phase boundary depends on the system size (here
    given for N=800 which is close to the TD limit).
    Alternatively, one can always compute the gap as the difference between the
    fifth eigenstate and the grounds state since in the TD limit the continuum
    level spacing vanishes.
    """
    if (lamX==0.0 && g<=1.0) || (lamX==0.1 && g<=0.937)
        psi0_init = randomMPS(sites,psi0_bonddim)
        psi1_init = randomMPS(sites,psi0_bonddim)
        psi2_init = randomMPS(sites,psi0_bonddim)
        psi3_init = randomMPS(sites,psi0_bonddim)
        psi4_init = randomMPS(sites,psi0_bonddim)
        if lamAD == 0.0
            energy0, psi0 = dmrg(H,psi0_init;
                            nsweeps, maxdim, cutoff, outputlevel=0)
            energy1, psi1 = dmrg(H,[psi0],psi1_init;
                            nsweeps,maxdim,cutoff, outputlevel=0, weight=weight)
            energy2, psi2 = dmrg(H,[psi0,psi1],psi2_init;
                            nsweeps,maxdim,cutoff, outputlevel=0, weight=weight)
            energy3, psi3 = dmrg(H,[psi0,psi1,psi2],psi3_init;
                            nsweeps,maxdim,cutoff, outputlevel=0, weight=weight)
            energy4, psi4 = dmrg(H,[psi0,psi1,psi2,psi3],psi4_init;
                            nsweeps,maxdim,cutoff, outputlevel=0, weight=weight)
        else
            energy0, psi0 = dmrg(H,psi0_init;
                    nsweeps, maxdim, cutoff, outputlevel=0, ishermitian=false)
            energy1, psi1 = dmrg(H,[psi0],psi1_init;
                    nsweeps,maxdim,cutoff, outputlevel=0, weight=weight,
                    ishermitian=false)
            energy2, psi2 = dmrg(H,[psi0,psi1],psi2_init;
                    nsweeps,maxdim,cutoff, outputlevel=0, weight=weight,
                    ishermitian=false)
            energy3, psi3 = dmrg(H,[psi0,psi1,psi2],psi3_init;
                    nsweeps,maxdim,cutoff, outputlevel=0, weight=weight,
                    ishermitian=false)
            energy4, psi4 = dmrg(H,[psi0,psi1,psi2,psi3],psi4_init;
                    nsweeps,maxdim,cutoff, outputlevel=0, weight=weight,
                    ishermitian=false)
        end
        gap = max(
                energy1-energy0,
                energy2-energy0,
                energy3-energy0,
                energy4-energy0
                )
    end

    if (lamX==0.0 && g>=1.0) || (lamX==0.1 && g>=0.937)
        psi0_init = randomMPS(sites,psi0_bonddim)
        psi1_init = randomMPS(sites,psi0_bonddim)
        if lamAD == 0.0
            energy0, psi0 = dmrg(H,psi0_init;
                            nsweeps, maxdim, cutoff, outputlevel=0)
            energy1, psi1 = dmrg(H,[psi0],psi1_init;
                            nsweeps,maxdim,cutoff, outputlevel=0, weight=weight)
        else
            energy0, psi0 = dmrg(H,psi0_init;
                            nsweeps, maxdim, cutoff, outputlevel=0,
                            ishermitian=false)
            energy1, psi1 = dmrg(H,[psi0],psi1_init;
                            nsweeps,maxdim,cutoff, outputlevel=0, weight=weight,
                            ishermitian=false)
        end
        gap = energy1-energy0
    end

    return gap

end


function dmrg_correlation_function(
    N :: Int,
    interaction_sign :: String,
    g :: Float64,
    lamX :: Float64,
    lamY :: Float64,
    lamZ :: Float64,
    lamXX :: Float64,
    lamYY :: Float64,
    lamZZ :: Float64,
    lamAD :: Float64,
    nsweeps :: Int,
    maxdim,
    cutoff,
    psi0_bonddim :: Int
)
    """
    Computes connected ZZ correlation function using DMRG.

    Parameters
    ----------
    N : Int
        The total number of sites on the ladder.
    interaction_sign : String
        Interaction along the legs of the ladder.
        Relevant options: FM or AFM
    g : Float64
        Transverse field.
    lamX : Float64
        XX-type interleg coupling (induced by X noise in the ITE).
    lamY : Float64
        YY-type interleg coupling (induced by Y noise in the ITE).
    lamZ : Float64
        ZZ-type interleg coupling (induced by Z noise in the ITE).
    lamXX : Float64
        XXXX-type plaquette interleg coupling (induced by XX noise in the ITE).
    lamYY : Float64
        YYYY-type plaquette interleg coupling (induced by YY noise in the ITE).
    lamZZ : Float64
        ZZZZ-type plaquette interleg coupling (induced by ZZ noise in the ITE).
    lamAD : Float64
        AD noise induced interleg coupling.
    nsweeps : Int
        The number of DMRG sweeps.
    maxdim :
        The maximum DMRG bod dimension.
    cutoff :
        DMRG cutoff.
    psi0_bonddim : Int
        Bond dimension of a random initial MPS in DMRG.
    """
    sites = siteinds("S=1/2",N)

    "Setting a spin ladder Hamiltonian MPO."
    os = OpSum()
    for j=1:2:N-2   # upper leg of the ladder
        if interaction_sign=="FM"
            os += -1,"Z",j,"Z",j+2
        end
        if interaction_sign=="AFM"
            os += +1,"Z",j,"Z",j+2
        end
        os += g,"X",j
    end
    os += g,"X",N-1
    for j=2:2:N-2   # lower leg of the ladder
        if interaction_sign=="FM"
            os += -1,"Z",j,"Z",j+2
        end
        if interaction_sign=="AFM"
            os += +1,"Z",j,"Z",j+2
        end
        os += g,"X",j
    end
    os += g,"X",N
    for j=1:2:N-1   # interleg coupling
        os += -lamX,"X",j,"X",j+1
        os += lamY,"Y",j,"Y",j+1
        os += -lamZ,"Z",j,"Z",j+1
    end
    for j=1:2:N-3   # interleg coupling
        os += -lamXX,"X",j,"X",j+1,"X",j+2,"X",j+3
        os += -lamYY,"Y",j,"Y",j+1,"Y",j+2,"Y",j+3
        os += -lamZZ,"Z",j,"Z",j+1,"Z",j+2,"Z",j+3
    end
    for j=1:N   # AD-noise induced term
        os += -lamAD,"Z",j
    end
    for j=1:2:N-1   # AD-noise induced terms
        os += -lamAD,"X",j,"X",j+1
        os += lamAD,"Y",j,"Y",j+1
        os += -im*lamAD,"X",j,"Y",j+1
        os += -im*lamAD,"Y",j,"X",j+1
    end
    H = MPO(os, sites)

    "Setting a sum_i Z_iZ_{i+1} MPO along the rungs of the ladder."
    os_obs_t = OpSum()
    for j=1:2:N-1
        os_obs_t += "Z",j,"Z",j+1
    end
    obs_t = MPO(os_obs_t, sites)

    """
    DMRG routine computing the ground state of the spin ladder.
    The routine is performed multiple times with a random initial state until
    the convergence criteria are met or the maximum number of iterations is
    reached.
    This is needed because sometimes DMRG can converge to a state within the
    ground state manifold that has a very small overlap with the |1>> state.
    The small overlap can lead to numerical errors.
    """
    for i in 1:100
        psi0 = randomMPS(sites, psi0_bonddim)
        if lamAD == 0.0
            energy_l, psi_l = dmrg(
                H,psi0;
                nsweeps,
                maxdim,
                cutoff,
                outputlevel=0,
                ishermitian=true
            )
        else
            "The Hamiltonian is non-Hermitian if the AD noise is present."
            energy_l, psi_l = dmrg(
                H,psi0;
                nsweeps,
                maxdim,
                cutoff,
                outputlevel=0,
                ishermitian=false
            )
        end
        global energy = energy_l
        global psi = psi_l
        """
        The first criterion ensures that the overlap between the ground state
        and the |1>> state is nonzero, i.e. that DMRG converged to the correct
        symmetry-broken state.
        The second criterion is added for the case of a disordered ground state.
        """
        if (real(inner(psi_l',obs_t,psi_l)) > 0 ||
            abs(real(inner(psi_l',obs_t,psi_l))) < 0.001)
            break
        end
    end

    "Setting an MPS for the |1>> state."
    A = [1 0
        0 1]
    id_efd_mps = MPS(sites,"0")
    for i in 1:div(N,2)
        Bell_MPS = MPS(A,sites[2*i-1:2*i],maxdim=2)
        id_efd_mps[2*i-1]=Bell_MPS[1]
        id_efd_mps[2*i]=Bell_MPS[2]
    end

    """
    Calculating connected ZZ correlation function over a range of distance given
    the obtained ground state of the spin ladder.
    The points at which the CF is computed are situated simmetrically with
    respect to the center of the Ising ladder.
    """
    CF=[]
    for i in 1:100
        os_obs_CF = OpSum()

        starts = 400-i
        ends = 400+i

        os_obs_CF += "Z",starts,"Z",ends
        obs_CF = MPO(os_obs_CF,sites)

        os_obs_CF1 = OpSum()
        os_obs_CF1 += "Z",starts
        obs_CF1 = MPO(os_obs_CF1,sites)

        os_obs_CF2 = OpSum()
        os_obs_CF2 += "Z",ends
        obs_CF2 = MPO(os_obs_CF2,sites)

        obs_CF_expectation = inner(id_efd_mps',obs_CF,psi)/inner(id_efd_mps,psi)
        obs_CF1_expectation = inner(id_efd_mps',obs_CF1,psi)/inner(id_efd_mps,psi)
        obs_CF2_expectation = inner(id_efd_mps',obs_CF2,psi)/inner(id_efd_mps,psi)

        append!(CF, obs_CF_expectation - obs_CF1_expectation * obs_CF2_expectation)
    end
    return CF
end
