import scipy.sparse as sp
import scipy.sparse.linalg as lg
import numpy as np
import warnings
kr = sp.kron

class State:
    def __init__(self, qubit_number, ini='0'):
        self.__qnum = qubit_number
        self.__ini = ini
        self.reset() # initialization of the state vector
        self.__gates = Gates(qubit_number)

    def load_lefthandside(self):
        if not hasattr(self, 'lhs'):
            self.lhs = sp.identity(2**self.__qnum, dtype='complex', format='csr')
        else:
            warnings.warn('lefthandside attribute already loaded. Ignoring function call.')

    def load_center_matrix(self, ini_matrix):
        if not hasattr(self, 'center_matrix'):
            self.center_matrix = ini_matrix.copy()
            self.__center_matrix_ini = ini_matrix.copy()
        else:
            warnings.warn('center_matrix attribute already loaded. Ignoring function call.')

    def reset(self):
        # reset vec
        if self.__ini == '0':
            self.vec = np.zeros(2**self.__qnum, dtype='complex')
            self.vec[0] = 1.
        elif self.__ini == '+':
            self.vec = 2.**(-.5*self.__qnum) * np.ones(2**self.__qnum, dtype='complex')
        else:
            raise ValueError('Initialization format {} not understood.'.format(self.__ini))
        # reset lhs
        if hasattr(self, 'lhs'):
            self.lhs = sp.identity(2**self.__qnum, dtype='complex', format='csr')
        # reset center_matrix
        if hasattr(self, 'center_matrix'):
            self.center_matrix = self.__center_matrix_ini.copy()

    def xrot(self, angle, i):
        self.vec = np.sin(.5*angle) * self.__gates.xrot[i].dot(self.vec) + \
            np.cos(.5*angle) * self.vec

    def dxrot(self, angle, i):
        '''Derivative of xrot.'''
        self.vec = .5*np.cos(.5*angle) * self.__gates.xrot[i].dot(self.vec) - \
            .5*np.sin(.5*angle) * self.vec

    def x_summed(self):
        '''Multiply the sum of all x-Paulis (incl. -1.j). For derivatives.'''
        self.vec = self.__gates.x_summed.dot(self.vec)

    def yrot(self, angle, i):
        self.vec = np.sin(.5*angle) * self.__gates.yrot[i].dot(self.vec) + \
            np.cos(.5*angle) * self.vec

    def dyrot(self, angle, i):
        '''Derivative of yrot.'''
        self.vec = .5*np.cos(.5*angle) * self.__gates.yrot[i].dot(self.vec) - \
            .5*np.sin(.5*angle) * self.vec

    def zrot(self, angle, i):
        self.vec = np.exp(-.5j*angle) * self.__gates.zrot_pos[i] * self.vec + \
            np.exp(.5j*angle) * self.__gates.zrot_neg[i] * self.vec

    def dzrot(self, angle, i):
        '''Derivative of zrot.'''
        self.vec = -.5j * np.exp(-.5j*angle) * self.__gates.zrot_pos[i] * self.vec + \
            .5j * np.exp(.5j*angle) * self.__gates.zrot_neg[i] * self.vec

    def cnot(self, i, j):
        self.vec = self.__gates.cnots[i, j].dot(self.vec)

    def cnot_ladder(self, stacking):
        '''Applies a one-dimensional ladder of CNOT gates

        Args:
            stacking (int): The even to uneven qubits are coupled first (i.e. 0-1,
                2-3, ... and 1-2, 3-4, ... in the second layer) for stacking=0.
                For stacking=1 the order is reversed.
        '''
        self.vec = self.__gates.cnot_ladder[stacking].dot(self.vec)

    def exp_ham_classical(self, angle):
        '''Rotate around a classical Hamiltonian, as done in QAOA.'''
        self.vec *= np.exp(-1.j * angle * self.__gates.classical_ham)

    def exp_ham_classical_component(self, angle, i):
        '''Rotate around a classical Hamiltonian component, as done in QAOA.'''
        self.vec *= np.exp(-1.j * angle * self.__gates.classical_ham_components[i])

    def ham_classical(self):
        '''Multiply a classical Hamiltonian (incl. -1.j) with the state. For derivatives.'''
        self.vec *= -1.j * self.__gates.classical_ham

    def custom_gate(self, key):
        self.vec = self.__gates.custom[key].dot(self.vec)

    def multiply_matrix(self, matrix):
        self.vec = matrix.dot(self.vec)

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

    def add_custom_gate(self, key, matrix):
        self.custom[key] = matrix
        return self

    def add_xrots(self):
        self.xrot = np.ndarray(self.qnum, dtype=sp.csr_matrix)
        for i in range(self.qnum):
            self.xrot[i] = -1.j * kr(
                kr(Gates.__id(i), Gates.__x),
                Gates.__id(self.qnum-i-1),
                format='csr'
            )
        return self

    def add_x_summed(self):
        '''All x gates summed up. I.e. -1.j * (x_1 + x_2 + ... + x_qnum).'''
        self.x_summed = sp.coo_matrix((2**self.qnum, 2**self.qnum), dtype='complex').asformat('csr')
        for i in range(self.qnum):
            self.x_summed += .5 * self.xrot[i]
        return self

    def add_yrots(self):
        self.yrot = np.ndarray(self.qnum, dtype=sp.csr_matrix)
        for i in range(self.qnum):
            self.yrot[i] = -1.j * kr(
                kr(Gates.__id(i), Gates.__y),
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
            raise ValueError((
                'CNOT gates in a ladder structure with periodic boundaries',
                'are ambiguous for uneven number of qubits.'
            ))
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

    def add_classical_ham(self, observable, include_individual_components=False):
        '''Creates a the full vector representing an observable, that is purely classical.

        Args:
            observable (Observable): An observable object.
        '''
        kr = np.kron # we operate on dense 1-d vectors
        observable.check_observable(
            known_keys=['z', 'zz'],
            warning='Non-classical observable component found. Only \'z\' and \'zz\' are accepted in this method.'
        )
        # build gate
        z = np.array([1., -1.])
        id = lambda i: np.full(2**i, 1.)
        self.classical_ham = np.zeros(2**self.qnum, dtype='double')
        component_list = []
        if 'z' in observable.info:
            for i, weight in enumerate(self.info['z']):
                if weight != None:
                    component = weight * kr(id(i), kr(z, id(self.qnum-i-1)))
                    self.classical_ham += component
                    if include_individual_components:
                        component_list.append(component)
        if 'zz' in observable.info:
            for i in range(self.qnum):
                for j in range(i+1, self.qnum):
                    if observable.info['zz'][i, j] != None:
                        component = observable.info['zz'][i, j] * kr(
                            id(i),
                            kr(z, kr(id(j-i-1), kr(z, id(self.qnum-j-1))))
                        )
                        self.classical_ham += component
                        if include_individual_components:
                            component_list.append(component)
        if include_individual_components:
            self.classical_ham_components = np.array(component_list)
        return self

    def __cnot(self, i, j):
        '''Controlled NOT gate. First argument is the control qubit.'''
        if i < j and j < self.qnum:
            out1 = kr(Gates.__id(i), Gates.__proj0)
            out1 = kr(out1, Gates.__id(self.qnum-i-1))

            out2 = kr(Gates.__id(i), Gates.__proj1)
            out2 = kr(out2, Gates.__id(j-i-1))
            out2 = kr(out2, Gates.__x)
            out2 = kr(out2, Gates.__id(self.qnum-j-1))
        elif i > j and i < self.qnum:
            out1 = kr(Gates.__id(i), Gates.__proj0)
            out1 = kr(out1, Gates.__id(self.qnum-i-1))

            out2 = kr(Gates.__id(j), Gates.__x)
            out2 = kr(out2, Gates.__id(i-j-1))
            out2 = kr(out2, Gates.__proj1)
            out2 = kr(out2, Gates.__id(self.qnum-i-1))
        else:
            raise ValueError('Invalid CNOT indecies {} and {}, for {} qubits.'.format(i, j, self.qnum))
        return (out1 + out2).asformat('csr')
