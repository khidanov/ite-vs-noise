

"function to compute the ground state of a spin ladder Hamiltonian and associated Binder cumulant corresponding to the mixed steady state of the noisy ITE"

function dmrg_binder(N,interaction_sign,g,lamX,lamY,lamZ,lamXX,lamYY,lamZZ,lamAD,nsweeps,maxdim,cutoff,psi0_bonddim)
    """
        N -- total number of sites on the ladder
        interaction_sign -- interaction along the legs of the ladder, FM or AFM
        g -- transverse field
        lamX -- XX-type interleg coupling (induced by X noise in the ITE)
        lamY -- YY-type interleg coupling (induced by Y noise in the ITE)
        lamZ -- ZZ-type interleg coupling (induced by Z noise in the ITE)
        lamXX -- XXXX-type plaquette interleg coupling (induced by XX noise in the ITE)
        lamYY -- YYYY-type plaquette interleg coupling (induced by YY noise in the ITE)
        lamZZ -- ZZZZ-type plaquette interleg coupling (induced by ZZ noise in the ITE)
        lamAD -- AD noise induced interleg coupling
        nsweeps -- number of DMRG sweeps
        maxdim -- maximum DMRG bod dimension
        cutoff -- DMRG cutoff
        psi0_bonddim -- bond dimension of a random initial MPS in DMRG
    """
    sites = siteinds("S=1/2",N)

    "setting a magnetization on the upper ladder (squared) MPO"
    os_mag2 = OpSum()
    for i=1:2:N-1, j=1:2:N-1
        if interaction_sign=="FM"
            os_mag2 += "Z",i,"Z",j
        end
        if interaction_sign=="AFM"
            os_mag2 += (-1)^((i+1)/2+(j+1)/2),"Z",i,"Z",j
        end
    end
    mag2 = MPO(os_mag2,sites)

    "setting a spin ladder Hamiltonian MPO"
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
    H = MPO(os,sites)

    "setting a sum_i Z_iZ_{i+1} MPO along the rungs of the ladder"
    os_obs_t = OpSum()
    for j=1:2:N-1
        os_obs_t += "Z",j,"Z",j+1
    end
    obs_t = MPO(os_obs_t,sites)

    "DMRG routine computing the ground state of the spin ladder. The routine is performed multiple times with a random initial state until the convergence criteria are met or the maximum number of iterations is reached.
    This is needed because sometimes DMRG can converge to a state within the ground state manifold that has a very small overlap with the |1>> state. The small overlap can lead to numerical errors."
    for i in 1:100
        psi0 = randomMPS(sites,psi0_bonddim)
        if lamAD == 0.0
            energy_l, psi_l = dmrg(H,psi0; nsweeps, maxdim, cutoff, outputlevel=0, ishermitian=true)
        else
            energy_l, psi_l = dmrg(H,psi0; nsweeps, maxdim, cutoff, outputlevel=0, ishermitian=false)    # the Hamiltonian is non-Hermitian if the AD noise is present
        end
        global energy = energy_l
        global psi = psi_l
        "The first criterion ensures that the overlap between the ground state and the |1>> state is nonzero, i.e. that DMRG converged to the correct symmetry-broken state. The second criterion is added for the case of a disordered ground state"
        if real(inner(psi_l',obs_t,psi_l)) > 0 || abs(real(inner(psi_l',obs_t,psi_l))) < 0.001
            break
        end
    end

    "setting an MPS for the |1>> state"
    A = [1 0
        0 1]
    id_efd_mps = MPS(sites,"0")
    for i in 1:div(N,2)
        Bell_MPS = MPS(A,sites[2*i-1:2*i],maxdim=2)
        id_efd_mps[2*i-1]=Bell_MPS[1]
        id_efd_mps[2*i]=Bell_MPS[2]
    end

    "calculating the mixed-state expectation value of <m^4> and <m^2> using the doubled space formalism"
    avg4 = inner(mag2,id_efd_mps,mag2,psi)/inner(id_efd_mps,psi)
    avg2 = inner(id_efd_mps',mag2,psi)/inner(id_efd_mps,psi)

    "outputting the Binder cumulant and the ground state energy"
    return real(3/2 - 1/2*avg4/avg2^2), energy
end




"Function to compute the ground states of a spin ladder Hamiltonian with and without noise-induced couplings, and evaluate the overlap beetween the two ground states.
This overlap corresponds to the fidelity between the steady states of the noiseless and the noisy ITE.
For the smaller system sizes that we've checked (N=50,100,200) and despite the symeetry breaking fields added at the ends of the ladder, the DMRG still sometimes picks ground states for H0 and H from different SB sectors, which results in a very small fidelity.
In this case, the DMRG calculation is needed to be performed again with a different initial state. This issue seems to disappear for larger system sizes (N=400,800)"

function dmrg_fidelity(N,interaction_sign,g,lamX,lamY,lamZ,lamXX,lamYY,lamZZ,lamAD,lamSBends,nsweeps,maxdim,cutoff,psi0_bonddim)
    """
        N -- total number of sites on the ladder
        interaction_sign -- interaction along the legs of the ladder, FM or AFM
        g -- transverse field
        lamX -- XX-type interleg coupling (induced by X noise in the ITE)
        lamY -- YY-type interleg coupling (induced by Y noise in the ITE)
        lamZ -- ZZ-type interleg coupling (induced by Z noise in the ITE)
        lamXX -- XXXX-type plaquette interleg coupling (induced by XX noise in the ITE)
        lamYY -- YYYY-type plaquette interleg coupling (induced by YY noise in the ITE)
        lamZZ -- ZZZZ-type plaquette interleg coupling (induced by ZZ noise in the ITE)
        lamAD -- AD noise induced interleg coupling
        lamSBends -- symmetry-breaking field added at the ends of the spin ladder to ensure that the DMRG picks a consistent summetry-broken grounds state
        nsweeps -- number of DMRG sweeps
        maxdim -- maximum DMRG bod dimension
        cutoff -- DMRG cutoff
        psi0_bonddim -- bond dimension of a random initial MPS in DMRG
    """
    sites = siteinds("S=1/2",N)

    "setting a spin ladder Hamiltonian MPO"
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

    H0 = MPO(os,sites)    # creating MPO for the Hamiltonian without the noise-induced coupling

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
    H = MPO(os,sites)    # creating MPO for the Hamiltonian with the noise-induced coupling

    "setting a sum_i Z_iZ_{i+1} MPO along the rungs of the ladder"
    os_obs_t = OpSum()
    for j=1:2:N-1
        os_obs_t += "Z",j,"Z",j+1
    end
    obs_t = MPO(os_obs_t,sites)

    "DMRG routine computing the ground state of H0. The routine is performed multiple times with a random initial state until the convergence criteria are met or the maximum number of iterations is reached.
    This is needed because sometimes DMRG can converge to a state within the ground state manifold that has a very small overlap with the |1>> state. The small overlap can lead to numerical errors."
    for i in 1:100
        psi0 = randomMPS(sites,psi0_bonddim)
        if lamAD == 0.0
            energy_l, psi_l = dmrg(H0,psi0; nsweeps, maxdim, cutoff, outputlevel=0, ishermitian=true)
        else
            energy_l, psi_l = dmrg(H0,psi0; nsweeps, maxdim, cutoff, outputlevel=0, ishermitian=false)    # the Hamiltonian is non-Hermitian if the AD noise is present
        end
        global energy_gs = energy_l
        global psi_gs = psi_l
        "The first criterion ensures that the overlap between the ground state and the |1>> state is nonzero, i.e. that DMRG converged to the correct symmetry-broken state. The second criterion is added for the case of a disordered ground state"
        if real(inner(psi_l',obs_t,psi_l)) > 0 || abs(real(inner(psi_l',obs_t,psi_l))) < 0.001
            break
        end
    end

    "DMRG routine computing the ground state of H. The routine is performed multiple times with a random initial state until the convergence criteria are met or the maximum number of iterations is reached.
    This is needed because sometimes DMRG can converge to a state within the ground state manifold that has a very small overlap with the |1>> state. The small overlap can lead to numerical errors."
    for i in 1:100
        psi0 = randomMPS(sites,psi0_bonddim)
        if lamAD == 0.0
            energy_l, psi_l = dmrg(H,psi0; nsweeps, maxdim, cutoff, outputlevel=0, ishermitian=true)
        else
            energy_l, psi_l = dmrg(H,psi0; nsweeps, maxdim, cutoff, outputlevel=0, ishermitian=false)    # the Hamiltonian is non-Hermitian if the AD noise is present
        end
        global energy = energy_l
        global psi = psi_l
        "The first criterion ensures that the overlap between the ground state and the |1>> state is nonzero, i.e. that DMRG converged to the correct symmetry-broken state. The second criterion is added for the case of a disordered ground state"
        if real(inner(psi_l',obs_t,psi_l)) > 0 || abs(real(inner(psi_l',obs_t,psi_l))) < 0.001
            break
        end
    end

    "outputting the overlap between the two ground states"
    return abs(inner(psi_gs',psi))
end
