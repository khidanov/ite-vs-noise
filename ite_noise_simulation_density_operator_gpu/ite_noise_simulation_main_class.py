
"""
class and functions for performing a GPU-assisted direct density matrix simulation of a noisy ITE and extracting the Binder cumulant of the steady state
CuPy (https://cupy.dev) is used as the GPU-accelerated computing library
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import identity
from scipy.sparse.linalg import expm
import cupy as cp
import cupyx as cpx
import random
import pickle
from joblib import Parallel, delayed

'function flipping a bit i in a binary n of length L'
def Flip(L, n, i):
    return n^(1<<(L-(i+1)))

'function converting a binary string into an int'
def bits2int(s):
    return int(''.join(s),2)

'function creates an array of L X-Pauli matrices associated with each qubit'
def X_func_cp(L):
    X=[]
    col = cp.arange(2**L)
    for i in range(L):
        row = cp.asarray(Flip(L,col,i))
        X.append(cpx.scipy.sparse.csr_matrix((cp.full(len(row),1), (row, col)), shape=(2**L, 2**L),dtype='float32'))
    return X

'function creates an array of L Z-Pauli matrices associated with each qubit'
def Z_func_cp(L):
    Z = [] # List of Z_i, i.e. Z_i = Z[i-1]
    row = cp.arange(2**L)
    for i in range(L):
        data = 1-2*(((row[:,None] & cp.asarray(1 << np.arange(L)))) > 0).astype(int)[:,L-i-1]
        Z.append(cpx.scipy.sparse.csr_matrix((data, (row, row)), shape=(2**L, 2**L), dtype='float32'))
    return Z

'function creates an array of L Y-Pauli matrices associated with each qubit'
def Y_func_cp(L):
    X = X_func_cp(L)
    Z = Z_func_cp(L)
    Y = [1j * X[i].dot(Z[i]) for i in range(L)] # List of Y_i, i.e. Y_i = Y[i-1]
    return Y

'function creates an array of nearest-neighbor ZZ-Pauli matrices'
def ZZ_func_cp(L,Z):
    ZZ = [Z[i].dot(Z[(i+1)%L]) for i in range(L)]
    return ZZ

class ite_noise_cp:
    """
    dtau is imaginary Trotter step size
    g is transverse field strength assuming Ising coupling J=1
    """
    def __init__(self, L=10):
        self.L = L

        self.Id = cpx.scipy.sparse.identity(2**self.L, dtype='float32', format='csr')

        self.X = X_func_cp(self.L)
        self.Z = Z_func_cp(self.L)
        self.Y = [1j * self.X[i].dot( self.Z[i]) for i in range(self.L)]
        self.ZZ = ZZ_func_cp(self.L,self.Z)

        'the magnetization operator and its powers needed to calculate Binder cumulants'
        self.mag = np.sum( [self.Z[i]/self.L for i in range(self.L)] , axis=0 )
        self.mag2 = (self.mag).dot( self.mag )
        self.mag4 = (self.mag2).dot( self.mag2 )

        'setting the initial state'

        'fully ferromagnetically ordered initial state'
        # col = cp.arange(2**self.L)
        # data = cp.zeros(2**self.L)
        # data[0] = 1.0
        # self.rho0 = cpx.scipy.sparse.csr_matrix((data, (col, col)), shape=(2**self.L, 2**self.L),dtype='float64')

        'fully antiferromagnetically ordered initial state'
        # col = cp.arange(2**self.L)
        # data = cp.zeros(2**self.L)
        # data[bits2int( '01'*(self.L//2) + '0'*(self.L%2)   )] = 1.0
        # self.rho0 = cpx.scipy.sparse.csr_matrix((data, (col, col)), shape=(2**self.L, 2**self.L),dtype='float64')

        'a disordered initial state'
        self.rho0 = cpx.scipy.sparse.csr_matrix( cp.ones((2**self.L,2**self.L))/2**self.L )

    'matrix exponent of a Pauli string'
    def exp_Pauli(self, coef, P):
        return np.cosh(coef) * self.Id - np.sinh(coef) * P

    'one run of the noisy ITE'
    def ite_fixed_run(self):

        rho = self.rho0

        U4_array = []

        for j in range(self.n_steps):
            'first applying all the ZZs layer by layer'
            rho = self.eZZ_layer_even.dot( rho.dot( cpx.scipy.sparse.csr_matrix.conjugate(cpx.scipy.sparse.csr_matrix.transpose(self.eZZ_layer_even))))
            rho = self.eZZ_layer_odd.dot( rho.dot( cpx.scipy.sparse.csr_matrix.conjugate(cpx.scipy.sparse.csr_matrix.transpose(self.eZZ_layer_odd))))

            'then applying all the Xs'
            for i in range(self.L):
                rho = self.eX[i].dot( rho.dot( self.eX[i]))

            'then appluing the noise'
            if self.noise_type == '1q_local_AD':
                for i in range(self.L):
                    rho = ((1+cp.sqrt(1-self.p))/2*self.Id + (1-cp.sqrt(1-self.p))/2*self.Z[i]).dot( rho.dot( (1+cp.sqrt(1-self.p))/2*self.Id + (1-cp.sqrt(1-self.p))/2*self.Z[i] ) ) + self.p/4*(self.X[i] + 1j*self.Y[i]).dot( rho.dot( self.X[i] - 1j*self.Y[i] ) )

            if self.noise_type == '2q_local_XX':
                for i in range(self.L-1):
                    rho = (1-self.p)*rho + self.p*self.X[i].dot(self.X[i+1].dot( rho.dot( self.X[i].dot(self.X[i+1]))))

            if self.noise_type == '2q_local_YY':
                for i in range(self.L-1):
                    rho = (1-self.p)*rho + self.p*self.Y[i].dot(self.Y[i+1].dot( rho.dot( self.Y[i].dot(self.Y[i+1]))))

            if self.noise_type == '2q_local_ZZ':
                for i in range(self.L-1):
                    rho = (1-self.p)*rho + self.p*self.Z[i].dot(self.Z[i+1].dot( rho.dot( self.Z[i].dot(self.Z[i+1]))))

            if self.noise_type == '1q_local_X':
                for i in range(self.L):
                    rho = (1-self.p)*rho + self.p*self.X[i].dot( rho.dot( self.X[i]))

            if self.noise_type == '1q_local_Y':
                for i in range(self.L):
                    rho = (1-self.p)*rho + self.p*self.Y[i].dot( rho.dot( self.Y[i]))

            if self.noise_type == '1q_local_Z':
                for i in range(self.L):
                    rho = (1-self.p)*rho + self.p*self.Z[i].dot( rho.dot( self.Z[i]))

            if self.noise_type == '1q_local_depolarizing':
                for i in range(self.L):
                    rho = (1-self.p)*rho + self.p/3*self.X[i].dot( rho.dot( self.X[i])) + self.p/3*self.Y[i].dot( rho.dot( self.Y[i])) + self.p/3*self.Z[i].dot( rho.dot( self.Z[i]))

            'normalization'
            rho = rho/cp.trace((rho).toarray()).item()

            'calculation of the expectation values of the observables'

            mag4_ev = cp.trace((self.mag4.dot( rho)).toarray()).item()
            mag2_ev = cp.trace((self.mag2.dot( rho)).toarray()).item()
            U4 = 3/2 - 1/2*mag4_ev/mag2_ev**2

            U4_array.append( U4 )

        return U4_array

    'running the same ite many times with differnet noise realizations and then averaging the results'
    def ite_many_runs(self, g, p, dtau, n_steps, noise_type, BC):
        """
        g -- transverese field
        p -- error probability
        dtau -- size of the ITE Trotter step
        n_steps -- number of ITE steps in the simulation
        BC -- boundary conditions
        """
        self.n_steps = n_steps
        self.p = p
        self.noise_type = noise_type

        'List of operators exp(-dtau*X) and exp(-dtau*g*ZZ), note that the interaction is FM as written'
        eZZ = [self.exp_Pauli(-dtau, self.ZZ[i]) for i in range(self.L)]
        self.eX = [self.exp_Pauli(-dtau * g, self.X[i]) for i in range(self.L)]

        'List of operators exp(-dtau*g*ZZ) corresponding to layers of gates (note that product of ZZs stays sparse, unlike product of Xs)'

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