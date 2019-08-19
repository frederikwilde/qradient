import scipy.sparse as sp
import scipy.sparse.linalg as lg
import numpy as np

class State:
    def __init__(self, qubit_number, gates=None, ini='0'):
        self.qnum = qubit_number
        self.__ini = ini
        self.reset() # initialization of the state vector
        self.gates = gates

    def xrot(self, angle, i):
        self.vec = np.sin(.5*angle) * self.gates.xrot[i].dot(self.vec) + \
            np.cos(.5*angle) * self.vec

    def dxrot(self, angle, i):
        '''Derivative of xrot.'''
        self.vec = .5*np.cos(.5*angle) * self.gates.xrot[i].dot(self.vec) - \
            .5*np.sin(.5*angle) * self.vec

    def yrot(self, angle, i):
        self.vec = np.sin(.5*angle) * self.gates.yrot[i].dot(self.vec) + \
            np.cos(.5*angle) * self.vec

    def dyrot(self, angle, i):
        '''Derivative of yrot.'''
        self.vec = .5*np.cos(.5*angle) * self.gates.yrot[i].dot(self.vec) - \
            .5*np.sin(.5*angle) * self.vec

    def zrot(self, angle, i):
        self.vec = np.exp(-.5j*angle) * self.gates.zrot_pos[i] * self.vec + \
            np.exp(.5j*angle) * self.gates.zrot_neg[i] * self.vec

    def dzrot(self, angle, i):
        '''Derivative of zrot.'''
        self.vec = -.5j * np.exp(-.5j*angle) * self.gates.zrot_pos[i] * self.vec + \
            .5j * np.exp(.5j*angle) * self.gates.zrot_neg[i] * self.vec

    def cnot(self, i, j):
        self.vec = self.gates.cnots[i, j].dot(self.vec)

    def cnot_ladder(self, stacking):
        '''Applies a one-dimensional ladder of CNOT gates

        Args:
            stacking (int): The even to uneven layers are coupled first (i.e. 0-1,
                2-3, ... and 1-2, 3-4, ... in the second stack) for stacking=0.
                For stacking=1 the order is reversed.
        '''
        self.vec = self.gates.cnot_ladder[stacking].dot(self.vec)

    def custom_gate(self, key):
        self.vec = self.gates.custom[key].dot(self.vec)

    def multiply_matrix(self, matrix):
        self.vec = matrix.dot(self.vec)

    def reset(self):
        if self.__ini == '0':
            self.vec = np.zeros(2**self.qnum, dtype='complex')
            self.vec[0] = 1.
        elif self.__ini == '+':
            self.vec = 2.**(-.5*self.qnum) * np.ones(2**self.qnum, dtype='complex')
        else:
            raise ValueError('Initialization format {} not understood.'.format(self.__ini))

    def norm_error(self):
        return 1. - np.linalg.norm(self.vec)

class Gates:
    __x = sp.coo_matrix([[0., 1.], [1., 0.]], dtype='complex')
    __y = sp.coo_matrix([[0., -1.j], [1.j, 0.]], dtype='complex')
    __id = lambda i: sp.identity(2**i, dtype='complex', format='coo')
    __proj0 = sp.coo_matrix([[1., 0.], [0., 0.]], dtype='complex')
    __proj1 = sp.coo_matrix([[0., 0.], [0., 1.]], dtype='complex')

    def __init__(self, qubit_number):
        self.qnum = qubit_number
        self.custom = {}

    def add_xrots(self):
        self.xrot = np.ndarray(self.qnum, dtype=sp.csr_matrix)
        for i in range(self.qnum):
            self.xrot[i] = -1.j * sp.kron(
                sp.kron(Gates.__id(i), Gates.__x),
                Gates.__id(self.qnum-i-1),
                format='csr'
            )
        return self

    def add_yrots(self):
        self.yrot = np.ndarray(self.qnum, dtype=sp.csr_matrix)
        for i in range(self.qnum):
            self.yrot[i] = -1.j * sp.kron(
                sp.kron(Gates.__id(i), Gates.__y),
                Gates.__id(self.qnum-i-1),
                format='csr'
            )
        return self

    def add_zrots(self):
        self.zrot_pos = np.zeros([self.qnum, 2**self.qnum], dtype='complex')
        self.zrot_neg = np.zeros([self.qnum, 2**self.qnum], dtype='complex')
        for i in range(self.qnum):
            interval = 2**(self.qnum-1-i)
            for j in range(0, 2**self.qnum, int(2*interval)):
                self.zrot_pos[i, j: j+interval] += 1.
                self.zrot_neg[i, j+interval: j+2*interval] += 1.
        return self

    def add_cnots(self, which):
        '''Adds CNOT gates into memory.

        Args:
            which (array[bool]): qnum by qnum matrix which specifies which CNOTs to add.
                Diagonal entries must be False.
        '''
        self.cnots = np.ndarray([self.qnum, self.qnum], dtype=sp.csr_matrix)
        for i in range(self.qnum):
            for j in range(self.qnum):
                if which[i, j]:
                    self.cnots[i, j] = self.__cnot(i, j)
        return self

    def add_cnot_ladder(self, periodic=False):
        # periodic ladders with uneven qubit number is ambiguous
        if periodic and (self.qnum%2 != 0):
            raise ValueError('CNOT gates in a ladder structure with periodic boundaries \
are ambiguous for uneven number of qubits.')
        # create 'which'-matrix
        which = np.full((self.qnum, self.qnum), False)
        for i in range(self.qnum-1):
            which[i, i+1] = True
        if periodic:
            which[self.qnum-1, 0] = True
        # create corresponding cnot gates
        cnots = np.ndarray([self.qnum, self.qnum], dtype=sp.csr_matrix)
        for i in range(self.qnum):
            for j in range(self.qnum):
                if which[i, j]:
                    cnots[i, j] = self.__cnot(i, j)
        # multiply them
        id = sp.identity(2**self.qnum, dtype='complex', format='csr')
        self.cnot_ladder = np.array([id, id])
        if periodic:
            upper = self.qnum
        else:
            upper = self.qnum - 1
        for i in range(0, upper, 2):
            self.cnot_ladder[0] = self.cnot_ladder[0].dot(cnots[i, (i+1)%self.qnum])
        for i in range(1, upper, 2):
            self.cnot_ladder[0] = self.cnot_ladder[0].dot(cnots[i, (i+1)%self.qnum])
            self.cnot_ladder[1] = self.cnot_ladder[1].dot(cnots[i, (i+1)%self.qnum])
        for i in range(0, upper, 2):
            self.cnot_ladder[1] = self.cnot_ladder[1].dot(cnots[i, (i+1)%self.qnum])
        return self

    def add_custom_gate(self, key, matrix):
        self.custom[key] = matrix
        return self

    def __cnot(self, i, j):
        '''Controlled NOT gate. First argument is the control qubit.'''
        if i < j and j < self.qnum:
            out1 = sp.kron(Gates.__id(i), Gates.__proj0)
            out1 = sp.kron(out1, Gates.__id(self.qnum-i-1))

            out2 = sp.kron(Gates.__id(i), Gates.__proj1)
            out2 = sp.kron(out2, Gates.__id(j-i-1))
            out2 = sp.kron(out2, Gates.__x)
            out2 = sp.kron(out2, Gates.__id(self.qnum-j-1))
        elif i > j and i < self.qnum:
            out1 = sp.kron(Gates.__id(i), Gates.__proj0)
            out1 = sp.kron(out1, Gates.__id(self.qnum-i-1))

            out2 = sp.kron(Gates.__id(j), Gates.__x)
            out2 = sp.kron(out2, Gates.__id(i-j-1))
            out2 = sp.kron(out2, Gates.__proj1)
            out2 = sp.kron(out2, Gates.__id(self.qnum-i-1))
        else:
            raise ValueError('Invalid CNOT indecies {} and {}, for {} qubits.'.format(i, j, self.qnum))
        return (out1 + out2).asformat('csr')
