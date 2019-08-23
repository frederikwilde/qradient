import scipy.sparse as sp
import scipy.sparse.linalg as lg
import numpy as np
import warnings
kr = sp.kron

class State:
    def __init__(self, qubit_number, gates=None, ini='0'):
        self.qnum = qubit_number
        self.__ini = ini # used in reset method
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
            stacking (int): The even to uneven qubits are coupled first (i.e. 0-1,
                2-3, ... and 1-2, 3-4, ... in the second layer) for stacking=0.
                For stacking=1 the order is reversed.
        '''
        self.vec = self.gates.cnot_ladder[stacking].dot(self.vec)

    def classical_ham(self, angle):
        '''Rotate around a classical Hamiltonian, as done in QAOA.'''
        gate = np.zeros(2**self.qnum, dtype='complex')
        for i in np.arange(len(self.gates.classical_ham_weights)):
            gate += np.exp(-1.j * angle * self.gates.classical_ham_weights[i]) * \
                self.gates.classical_ham_components[i]
        self.vec *= gate

    def custom_gate(self, key):
        self.vec = self.gates.custom[key].dot(self.vec)

    def multiply_matrix(self, matrix):
        self.vec = matrix.dot(self.vec)

    def reset(self, ini=None):
        if not ini == None:
            self.__ini = ini
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

    def add_classical_ham(self, classical_ham):
        self.classical_ham_weights = classical_ham[0]
        self.classical_ham_components = classical_ham[1]
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

class Observable:
    def __init__(self, qubit_number, observable):
        '''Takes a dictionary specifying the observable and builds the matrix.

        Args:
            qubit_number (int): Number of qubits.
            observable (dict): can contain the following fields:
                'x': a 1d-array of prefactors for single-qubit Pauli X. Choose None
                    for no matrix.
                'y': 1d-array for Pauli Y.
                'z': 1d-array for Pauli Z.
                'zz': a 2d-array as upper triangular matrix for two-qubit ZZ terms.
        '''
        self.qnum = qubit_number
        # FLAGS
        self.has_loaded_projectors = False
        # load observable
        self.info = observable
        self.__check_observable(known_keys=['x', 'y', 'z', 'zz'])
        self.matrix = sp.coo_matrix((2**self.qnum, 2**self.qnum), dtype='complex').asformat('csr')
        x = sp.csr_matrix([[0., 1.], [1., 0.]], dtype='complex')
        y = sp.csr_matrix([[0., -1.j], [1.j, 0.]], dtype='complex')
        z = sp.csr_matrix([[1., 0.], [0., -1.]], dtype='complex')
        # single-qubit components
        for identifier, matrix in [('x', x), ('y', y), ('z', z)]:
            if identifier in observable:
                for i, weight in enumerate(observable[identifier]):
                    if weight != None:
                        Observable.__weight_check(weight, identifier)
                        op = kr(kr(sp.identity(2**i), matrix), sp.identity(2**(self.qnum-i-1)))
                        self.matrix += weight * op
        # two-qubit components
        if 'zz' in observable:
            for i in range(self.qnum):
                for j in range(i+1):
                    if observable['zz'][i, j] != None:
                        raise ValueError((
                            'zz of observable should be a upper triangular {} '.format(self.qnum),
                            'by {} matrix. Diagonal and lower triangle should'.format(self.qnum),
                            'contain None\'s, not {}.'.format(observable['zz'][i, j])
                        ))
                for j in range(i+1, self.qnum):
                    if observable['zz'][i, j] != None:
                        Observable.__weight_check(observable['zz'][i, j], 'y')
                        op = kr(kr(sp.identity(2**i), z), sp.identity(2**(j-i-1)))
                        op = kr(kr(op, z), sp.identity(2**(self.qnum-j-1)))
                        self.matrix += observable['zz'][i, j] * op

    def load_projectors(self):
        if self.has_loaded_projectors: # in case the caller has not checked
            return None
        self.__check_observable(known_keys=['x', 'y', 'z', 'zz'])
        # construct pauli projectors
        projectors = []
        projector_weights = []
        x_proj_pos = sp.csr_matrix([[.5, .5], [.5, .5]], dtype='complex') # projects on the +1 subspace
        y_proj_pos = sp.csr_matrix([[.5, -.5j], [.5j, .5]], dtype='complex') # projects on the +1 subspace
        z_proj_pos = sp.csr_matrix([[1., 0.], [0., 0.]], dtype='complex') # projects on the +1 subspace
        z_proj_neg = sp.csr_matrix([[0., 0.], [0., 1.]], dtype='complex') # projects on the -1 subspace
        gates = Gates(self.qnum).add_zrots() # for the z projectors
        # read the content of observable
        x = self.info.get('x', np.full(self.qnum, None))
        y = self.info.get('y', np.full(self.qnum, None))
        z = self.info.get('z', np.full(self.qnum, None))
        zz = self.info.get('zz', np.full([self.qnum, self.qnum], None))
        for i in range(self.qnum):
            id1 = sp.identity(2**i, dtype='complex')
            id2 = sp.identity(2**(self.qnum-i-1), dtype='complex')
            if x[i] != None:
                projectors.append(kr(id1, kr(x_proj_pos, id2), format='csr'))
                projector_weights.append(x[i])
            if y[i] != None:
                projectors.append(kr(id1, kr(y_proj_pos, id2), format='csr'))
                projector_weights.append(y[i])
            if z[i] != None:
                projectors.append(sp.diags(gates.zrot_pos[i], format='csr'))
                projector_weights.append(z[i])
            for j in range(i+1, self.qnum):
                if zz[i, j] != None:
                    id3 = sp.identity(2**(j-i-1), dtype='complex')
                    id4 = sp.identity(2**(self.qnum-j-1), dtype='complex')
                    upup = kr(
                        id1,
                        kr(z_proj_pos, kr(id3, kr(z_proj_pos, id4))),
                        format='csr'
                    )
                    downdown = kr(
                        id1,
                        kr(z_proj_neg, kr(id3, kr(z_proj_neg, id4))),
                        format='csr'
                    )
                    projectors.append(upup + downdown)
                    projector_weights.append(zz[i, j])
        self.projectors = np.array(projectors)
        self.projector_weights = np.array(projector_weights)
        self.has_loaded_projectors = True

    def classical_to_gate(self):
        # make sure observable only contains z and zz
        kr = np.kron # we operate on dense 1-d vectors
        self.__check_observable(
            known_keys=['z', 'zz'],
            warning='Non-classical observable component found. Only \'z\' and \'zz\' are accepted.'
        )
        # build gate
        z = np.array([1., -1.])
        id = lambda i: np.full(2**i, 1.)
        gate = id(self.qnum)
        if 'z' in self.info:
            for i, weight in enumerate(self.info['z']):
                if weight != None:
                    gate += weight * kr(id(i), kr(z, id(self.qnum-i-1)))
        if 'zz' in self.info:
            for i in range(self.qnum):
                for j in range(i+1, self.qnum):
                    if self.info['zz'][i, j] != None:
                        gate += self.info['zz'][i, j] * kr(
                            id(i),
                            kr(z, kr(id(j-i-1), kr(z, id(self.qnum-j-1))))
                        )
        unique = np.unique(gate)
        if len(unique) > self.qnum**2:
            warnings.warn((
                'Observable contains many terms, gate decomposition might be ',
                'inefficient. {} different eigenvalues'.format(len(unique))
            ))
        gate_components = np.ndarray([len(unique), 2**self.qnum], dtype='complex')
        gate_weights = np.ndarray(len(unique), dtype='complex')
        for i, weight in enumerate(unique):
            gate_components[i] = (gate == weight) # binary string
            gate_weights[i] = weight
        return gate_weights, gate_components

    def __check_observable(self, known_keys, warning=None):
        '''Tests whether observable only contains known keys and throws a warning otherwise.'''
        for key in list(self.info.keys()):
            unknown = True
            for k in known_keys:
                if key == k:
                    unknown = False
            if unknown:
                if warning == None:
                    warnings.warn('Unknown element of observable {} will be ignored.'.format(key))
                else:
                    warnings.warn(warning)

    def __weight_check(weight, observable_component):
        if abs(weight) < 10.**-15:
            warnings.warn((
                'Weight in observable {} is zero or almost zero.'.format(observable_component),
                ' If you dont\'t want to include it, set it to None.'
            ))
