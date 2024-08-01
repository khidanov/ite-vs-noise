
"""
This module performs GPU-assisted direct density matrix simulation of a noisy
Trotterized ITE for 1D TFIM and extracts Binder cumulant.
Binder cumulant is used to identify phase trasnition in the steady state of the
noisy ITE via finite-size scaling.

Reference: https://arxiv.org/abs/2406.04285.

CuPy (https://cupy.dev) package is used for the GPU-accelerated computing.

Packages information:
---------------------
python >= 3.9
CuPy version = 12.1.0
CUDA compiler version = 10.2
numpy version = 1.26.2
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import identity
from scipy.sparse.linalg import expm
import cupy as cp
import cupyx as cpx
from typing import (
    List,
    Optional,
    Tuple,
    Union
)


def Flip(L, n, i):
    """
    Flips a bit with an index i in a binary representation (of length L) of an
    integer n.

    Parameters
    ----------
    L : int
        Length of a binary representation of integer n (with zeros added to the
        front of the binary string, if needed).
    n : int
        Integer, whose binary representation is subjected to bit flipping.
    i : int
        Index of a bit being flipped.

    Returns
    -------
    n_bitflipped : int
        Integer represeting the binary with the flipped bit.
    """
    n_bitflipped = n^(1<<(L-(i+1)))
    return n_bitflipped


def bits2int(s):
    """
    Converts a binary string into integer.

    Parameters
    ----------
    s : str
        Binary string to be converted.

    Returns
    -------
    n_int : int
        Integer representing the binary string s.
    """
    n_int = int(''.join(s),2)
    return n_int


def X_func_cp(L):
    """
    For L qubits, generates a list (of length L) of X Pauli matrices in the
    sparse matrix format indexed by a qubit number.

    Parameters
    ----------
    L : int
        Number of qubits.

    Returns
    -------
    X : List[cupyx.scipy.sparse.csr_matrix]
        List of X Pauli matrices.
    """
    X = []
    col = cp.arange(2**L)
    for i in range(L):
        row = cp.asarray(Flip(L,col,i))
        X.append(
            cpx.scipy.sparse.csr_matrix((cp.full(len(row),1), (row, col)),
            shape=(2**L, 2**L),dtype='float32')
        )
    return X


def Z_func_cp(L):
    """
    For L qubits, generates a list (of length L) of Z Pauli matrices in the
    sparse matrix format indexed by a qubit number.

    Parameters
    ----------
    L : int
        Number of qubits.

    Returns
    -------
    Z : List[cupyx.scipy.sparse.csr_matrix]
        List of Z Pauli matrices.
    """
    Z = []
    row = cp.arange(2**L)
    for i in range(L):
        data = 1-2*(
            (row[:,None] & cp.asarray(1 << np.arange(L))) > 0
        ).astype(int)[:,L-i-1]
        Z.append(
            cpx.scipy.sparse.csr_matrix((data, (row, row)),
            shape=(2**L, 2**L), dtype='float32')
        )
    return Z


def Y_func_cp(L):
    """
    For L qubits, generates a list (of length L) of Y Pauli matrices in the
    sparse matrix format indexed by a qubit number.

    Parameters
    ----------
    L : int
        Number of qubits.

    Returns
    -------
    Y : List[cupyx.scipy.sparse.csr_matrix]
        List of Y Pauli matrices.
    """
    X = X_func_cp(L)
    Z = Z_func_cp(L)
    Y = [1j * X[i].dot(Z[i]) for i in range(L)]
    return Y


def ZZ_func_cp(L, Z):
    """
    For L qubits, generates a list (of length L) of nearest-neighbor ZZ Pauli
    matrices in the sparse matrix format indexed by a qubit number.

    Parameters
    ----------
    L : int
        Number of qubits.

    Returns
    -------
    ZZ : List[cupyx.scipy.sparse.csr_matrix]
        List of ZZ Pauli matrices.
    """
    ZZ = [Z[i].dot(Z[(i+1)%L]) for i in range(L)]
    return ZZ


class ITE_noise_cp:
    """
    Class for performing GPU-assisted direct density matrix simulation of a
    noisy Trotterized ITE for 1D TFIM.
    The main method performing the simulation is ite_density_matrix.
    The Binder cumulant obtained through the simulations is then used to
    detemine phase transitions in the noisy model.

    Attributes
    ----------
    L : int
        Number of sites.
    Id : cpx.scipy.sparse.csr_matrix
        Identity matrix for the given Hibert space in the sparse CSR format.
    rho0 : cpx.scipy.sparse.csr_matrix
        Initial state.
    """
    def __init__(self, L = 10):
        self.L = L
        self.Id = cpx.scipy.sparse.identity(
            2**self.L, dtype='float32', format='csr'
        )
        self.X = X_func_cp(self.L)
        self.Z = Z_func_cp(self.L)
        self.Y = [1j * self.X[i].dot( self.Z[i]) for i in range(self.L)]
        self.ZZ = ZZ_func_cp(self.L,self.Z)

        """
        The magnetization operator and its powers are needed to calculate Binder
        cumulants.
        """
        self.mag = np.sum( [self.Z[i]/self.L for i in range(self.L)], axis = 0 )
        self.mag2 = (self.mag).dot( self.mag )
        self.mag4 = (self.mag2).dot( self.mag2 )

        'Setting the initial state:'

        'fully ferromagnetically ordered initial state'
        # col = cp.arange(2**self.L)
        # data = cp.zeros(2**self.L)
        # data[0] = 1.0
        # self.rho0 = cpx.scipy.sparse.csr_matrix(
        #     (data, (col, col)), shape=(2**self.L, 2**self.L), dtype='float64'
        # )

        'fully antiferromagnetically ordered initial state'
        # col = cp.arange(2**self.L)
        # data = cp.zeros(2**self.L)
        # data[bits2int( '01'*(self.L//2) + '0'*(self.L%2)   )] = 1.0
        # self.rho0 = cpx.scipy.sparse.csr_matrix(
        #     (data, (col, col)), shape=(2**self.L, 2**self.L), dtype='float64'
        # )

        'a disordered initial state'
        self.rho0 = cpx.scipy.sparse.csr_matrix(
            cp.ones((2**self.L,2**self.L))/2**self.L
        )


    def exp_Pauli(
        self,
        coef: float,
        P
    ):
        """
        Returns matrix exponential of a Pauli string.

        Parameters
        ----------
        P : cpx.scipy.sparse.csr_matrix
            Pauli string
        coef : float
            Coefficient under the exponent.
        """
        return np.cosh(coef) * self.Id - np.sinh(coef) * P


    def ite_fixed_run(self):
        """
        Performs a run of a noisy ITE simulation.
        """

        rho = self.rho0

        U4_list = []

        for j in range(self.n_steps):
            'first applying all the ZZs layer by layer'
            rho = self.eZZ_layer_even.dot(
                rho.dot(
                    cpx.scipy.sparse.csr_matrix.conjugate(
                        cpx.scipy.sparse.csr_matrix.transpose(
                            self.eZZ_layer_even
                        )
                    )
                )
            )
            rho = self.eZZ_layer_odd.dot(
                rho.dot(
                    cpx.scipy.sparse.csr_matrix.conjugate(
                        cpx.scipy.sparse.csr_matrix.transpose(
                            self.eZZ_layer_odd
                        )
                    )
                )
            )
            'then applying all the Xs'
            for i in range(self.L):
                rho = self.eX[i].dot( rho.dot( self.eX[i]))

            'then appluing the noise'
            if self.noise_type == '1q_local_AD':
                for i in range(self.L):
                    rho = (
                        (
                            (1+cp.sqrt(1-self.p))/2*self.Id +
                            (1-cp.sqrt(1-self.p))/2*self.Z[i]
                        ).dot(
                            rho.dot(
                                (1+cp.sqrt(1-self.p))/2*self.Id +
                                (1-cp.sqrt(1-self.p))/2*self.Z[i]
                            )
                        ) + self.p/4*(self.X[i] + 1j*self.Y[i]).dot(
                            rho.dot(self.X[i] - 1j*self.Y[i] )
                        )
                    )
            if self.noise_type == '2q_local_XX':
                for i in range(self.L-1):
                    rho = (1-self.p)*rho + self.p*self.X[i].dot(
                        self.X[i+1].dot(
                            rho.dot(
                                self.X[i].dot(self.X[i+1])
                            )
                        )
                    )
            if self.noise_type == '2q_local_YY':
                for i in range(self.L-1):
                    rho = (1-self.p)*rho + self.p*self.Y[i].dot(
                        self.Y[i+1].dot(
                            rho.dot( self.Y[i].dot(self.Y[i+1]))
                        )
                    )
            if self.noise_type == '2q_local_ZZ':
                for i in range(self.L-1):
                    rho = (1-self.p)*rho + self.p*self.Z[i].dot(
                        self.Z[i+1].dot(
                            rho.dot(self.Z[i].dot(self.Z[i+1]))
                        )
                    )
            if self.noise_type == '1q_local_X':
                for i in range(self.L):
                    rho = (1-self.p)*rho + self.p*self.X[i].dot(
                        rho.dot(self.X[i])
                    )
            if self.noise_type == '1q_local_Y':
                for i in range(self.L):
                    rho = (1-self.p)*rho + self.p*self.Y[i].dot(
                        rho.dot(self.Y[i])
                    )

            if self.noise_type == '1q_local_Z':
                for i in range(self.L):
                    rho = (1-self.p)*rho + self.p*self.Z[i].dot(
                        rho.dot(self.Z[i])
                    )

            if self.noise_type == '1q_local_depolarizing':
                for i in range(self.L):
                    rho = (
                        (1-self.p)*rho +
                        self.p/3*self.X[i].dot(rho.dot(self.X[i])) +
                        self.p/3*self.Y[i].dot(rho.dot(self.Y[i])) +
                        self.p/3*self.Z[i].dot(rho.dot(self.Z[i]))
                    )

            'normalization'
            rho = rho/cp.trace((rho).toarray()).item()

            'calculation of the expectation values of the observables'
            mag4_ev = cp.trace((self.mag4.dot( rho)).toarray()).item()
            mag2_ev = cp.trace((self.mag2.dot( rho)).toarray()).item()
            U4 = 3/2 - 1/2*mag4_ev/mag2_ev**2

            U4_list.append( U4 )

        return U4_list


    def ite_density_matrix(
        self,
        g: float = 0.1,
        p: float = 0.01,
        dtau: float = 0.1,
        n_steps: int = 200,
        noise_type: Optional[str] = '1q_local_X',
        BC: str = 'open'
    ):
        """
        Performs direct density matrix simulations of the Trotterized noisy ITE
        process for the 1D TFIM Hamiltonian.

        Parameters
        ----------
        g : float
            Transevrse field value.
        p : float
            Noise strength (probability of error occurence), 0<=p<=1.
        dtau : float
            Imaginary time Trotter step size.
        n_steps : int
            Number of imaginary time steps in the simulation.
            Number of steps should be chosen such that the steady state is
            reached.
        noise_type : Optional[str]
            Type of noise.
            Possible options: '1q_local_X', '1q_local_Y', '1q_local_Z',
            '1q_local_depolarizing', '1q_local_AD', '2q_local_XX',
            '2q_local_YY', '2q_local_ZZ'.
        BC : str
            Boundary conditions for the 1D TFIM.
            Possible options: 'open', 'periodic'.
            For better comparisons with DMRG results (in the limit of small
            imaginary-time Trotter step) BC='open' should be used.

        Returns
        -------
        U4_data : List[float]
            List of Binder cumulants computed along the noisy imaginary time
            process.
        """
        self.n_steps = n_steps
        self.p = p
        self.noise_type = noise_type

        """
        List of operators exp(-dtau*X) and exp(-dtau*g*ZZ),
        note that the interaction is FM as written
        """
        eZZ = [self.exp_Pauli(-dtau, self.ZZ[i]) for i in range(self.L)]
        self.eX = [self.exp_Pauli(-dtau * g, self.X[i]) for i in range(self.L)]

        """
        List of operators exp(-dtau*g*ZZ) corresponding to layers of gates
        (note that product of ZZs stays sparse, unlike product of Xs)
        """
        self.eZZ_layer_even = self.Id
        if BC == 'periodic':
            for i in range(self.L):
                if i % 2 == 0:
                    self.eZZ_layer_even = (self.eZZ_layer_even).dot(eZZ[i] )
        if BC == 'open':
            for i in range(self.L-1):
                if i % 2 == 0:
                    self.eZZ_layer_even = (self.eZZ_layer_even).dot(eZZ[i] )

        self.eZZ_layer_odd = self.Id
        if BC == 'periodic':
            for i in range(self.L):
                if i % 2 == 1:
                    self.eZZ_layer_odd = (self.eZZ_layer_odd).dot(eZZ[i] )
        if BC == 'open':
            for i in range(self.L-1):
                if i % 2 == 1:
                    self.eZZ_layer_odd = (self.eZZ_layer_odd).dot(eZZ[i] )

        U4_data = self.ite_fixed_run()

        return U4_data
