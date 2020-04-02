import scipy.sparse as sp
import numpy as np
import warnings



class State:
    '''
    Objects of this type can contain up to three attributes. A state vector
    is always contained. Additionally a left-hand-side matrix can be stored to
    track the reverse circuit as well as a center matrix where gates are applied
    to both sides (for sandwhiching with a state).

    For all these attributes the methods provided apply gates. Those gates must
    be loaded prior to application.

    Args:
        qubit_number (int): The system size.
        ini (str):
            The initial state, which is also the reference for the reset
            method.

    Attributes:
        vec (np.ndarray): The state vector
        lhs (scipy.sparse.csr_matrix):
            A left-hand-side matrix initialized to the identity. It will
            quickly become a dense matrix, but for multiplication with sparse
            gates it is stored as a sparse matrix.
        center_matrix (scipy.sparse.csr_matrix):
            A center matrix, where gates are applied on both sides
            simultaneously.
    '''
    def __init__(self, qubit_number, ini='0'):
        self.__qnum = qubit_number
        self.__ini = ini
        self.reset() # initialization of the state vector

    def activate_lefthandside(self):
        '''Initializes the left-hand-side attribute.
        If the attribute is already loaded, this call is ignored.'''
        if not hasattr(self, 'lhs'):
            self.lhs = sp.identity(2**self.__qnum, dtype='complex', format='csr')

    def activate_center_matrix(self):
        '''Initializes the center-matrix to an empty matrix.
        Use set_center_matrix to fill it.
        If the attribute is already loaded, this call is ignored.'''
        if not hasattr(self, 'center_matrix'):
            self.center_matrix = sp.csr_matrix((2**self.__qnum, 2**self.__qnum), dtype='complex')
            self.__center_matrix_ini = sp.csr_matrix((2**self.__qnum, 2**self.__qnum), dtype='complex')

    def set_center_matrix(self, matrix):
        '''Updates the center-matrix (if activated) to matrix.
        Also stores the matrix for the reset method.'''
        if not hasattr(self, 'center_matrix'):
            raise AttributeError('center_matrix is not initialized yet.')
        self.center_matrix = matrix.copy()
        self.__center_matrix_ini = matrix.copy()

    def reset(self):
        '''Resets the state vector and left-hand-side and center matrix, if
        they are activated.'''
        # reset vec
        if self.__ini == '0':
            self.vec = np.zeros(2**self.__qnum, dtype='complex')
            self.vec[0] = 1.
        elif self.__ini == '+':
            self.vec = 2.**(-.5*self.__qnum) * np.ones(2**self.__qnum, dtype='complex')
        else:
            raise ValueError('Invalid initialization format {}.'.format(self.__ini))
        # reset lhs
        if hasattr(self, 'lhs'):
            self.lhs = sp.identity(2**self.__qnum, dtype='complex', format='csr')
        # reset center_matrix
        if hasattr(self, 'center_matrix'):
            self.center_matrix = self.__center_matrix_ini.copy()

    ############################################################################
    # x rotations
    def load_xrots(self):
        self.__xrot = np.ndarray(self.__qnum, dtype=sp.csr_matrix)
        for i in range(self.__qnum):
            self.__xrot[i] = -1.j * _skr(
                _skr(_id(i), _x),
                _id(self.__qnum-i-1),
                format='csr'
            )

    def xrot(self, angle, i):
        self.vec = np.sin(.5*angle) * self.__xrot[i].dot(self.vec) + \
            np.cos(.5*angle) * self.vec

    def dxrot(self, angle, i):
        '''Derivative of xrot.'''
        self.vec = .5*np.cos(.5*angle) * self.__xrot[i].dot(self.vec) - \
            .5*np.sin(.5*angle) * self.vec

    def xrot_lhs(self, angle, i):
        warnings.warn('Not implemented.')

    def xrot_center_matrix(self, angle, i):
        warnings.warn('Not implemented.')

    ############################################################################
    # x rotation with all x (x_1 + x_2 + ...)
    def load_xrot_all(self):
        '''All x gates summed up. I.e. -1.j * (x_1 + x_2 + ... + x_qnum).'''
        temp_xrot = np.ndarray(self.__qnum, dtype=sp.csr_matrix)
        for i in range(self.__qnum):
            temp_xrot[i] = -1.j * _skr(
                _skr(_id(i), _x),
                _id(self.__qnum-i-1),
                format='csr'
            )
        self.__xrot_all = sp.coo_matrix((2**self.__qnum, 2**self.__qnum), dtype='complex').asformat('csr')
        for i in range(self.__qnum):
            self.__xrot_all += .5 * temp_xrot[i]
        del temp_xrot

    def xrot_all(self):
        '''Multiply the sum of all x-Paulis (incl. -1.j). For derivatives.'''
        self.vec = self.__xrot_all.dot(self.vec)

    def xrot_all_lhs(self, angle, i):
        warnings.warn('Not implemented.')

    def xrot_all_center_matrix(self, angle, i):
        warnings.warn('Not implemented.')

    ############################################################################
    # y rotations
    def load_yrots(self):
        self.__yrot = np.ndarray(self.__qnum, dtype=sp.csr_matrix)
        for i in range(self.__qnum):
            self.__yrot[i] = -1.j * _skr(
                _skr(_id(i), _y),
                _id(self.__qnum-i-1),
                format='csr'
            )

    def yrot(self, angle, i):
        self.vec = np.sin(.5*angle) * self.__yrot[i].dot(self.vec) + \
            np.cos(.5*angle) * self.vec

    def dyrot(self, angle, i):
        '''Derivative of yrot.'''
        self.vec = .5*np.cos(.5*angle) * self.__yrot[i].dot(self.vec) - \
            .5*np.sin(.5*angle) * self.vec

    def yrot_lhs(self, angle, i):
        warnings.warn('Not implemented.')

    def yrot_center_matrix(self, angle, i):
        warnings.warn('Not implemented.')

    ############################################################################
    # z rotations
    def load_zrots(self):
        self.__zrot_pos = np.zeros([self.__qnum, 2**self.__qnum], dtype='complex')
        self.__zrot_neg = np.zeros([self.__qnum, 2**self.__qnum], dtype='complex')
        for i in range(self.__qnum):
            interval = 2**(self.__qnum-1-i)
            for j in range(0, 2**self.__qnum, int(2*interval)):
                self.__zrot_pos[i, j: j+interval] += 1.
                self.__zrot_neg[i, j+interval: j+2*interval] += 1.

    def zrot(self, angle, i):
        self.vec = np.exp(-.5j*angle) * self.__zrot_pos[i] * self.vec + \
            np.exp(.5j*angle) * self.__zrot_neg[i] * self.vec

    def dzrot(self, angle, i):
        '''Derivative of zrot.'''
        self.vec = -.5j * np.exp(-.5j*angle) * self.__zrot_pos[i] * self.vec + \
            .5j * np.exp(.5j*angle) * self.__zrot_neg[i] * self.vec

    def zrot_lhs(self, angle, i):
        warnings.warn('Not implemented.')

    def zrot_center_matrix(self, angle, i):
        warnings.warn('Not implemented.')

    ############################################################################
    # CNOTs
    def load_cnots(self, which):
        '''Adds CNOT gates into memory.

        Args:
            which (array[bool]): qnum by qnum matrix which specifies which CNOTs to add.
                Diagonal entries must be False.
        '''
        self.__cnots = np.ndarray([self.__qnum, self.__qnum], dtype=sp.csr_matrix)
        for i in range(self.__qnum):
            for j in range(self.__qnum):
                if which[i, j]:
                    self.__cnots[i, j] = self.__cnot(i, j)

    def cnot(self, i, j):
        self.vec = self.__cnots[i, j].dot(self.vec)

    def cnot_lhs(self, i, j):
        warnings.warn('Not implemented.')

    def cnot_center_matrix(self, i, j):
        warnings.warn('Not implemented.')

    ############################################################################
    # CNOT ladder
    def load_cnot_ladder(self, periodic=False):
        # periodic ladders with uneven qubit number is ambiguous
        if periodic and (self.__qnum % 2 != 0):
            raise ValueError((
                'CNOT gates in a ladder structure with periodic boundaries',
                'are ambiguous for uneven number of qubits.'
            ))
        # create 'which'-matrix
        which = np.full((self.__qnum, self.__qnum), False)
        for i in range(self.__qnum-1):
            which[i, i+1] = True
        if periodic:
            which[self.__qnum-1, 0] = True
        # create corresponding cnot gates
        cnots = np.ndarray([self.__qnum, self.__qnum], dtype=sp.csr_matrix)
        for i in range(self.__qnum):
            for j in range(self.__qnum):
                if which[i, j]:
                    cnots[i, j] = self.__cnot(i, j)
        # multiply them
        id = sp.identity(2**self.__qnum, dtype='complex', format='csr')
        self.__cnot_ladder = np.array([id.copy(), id.copy()])
        if periodic:
            upper = self.__qnum
        else:
            upper = self.__qnum - 1
        for i in range(0, upper, 2):
            self.__cnot_ladder[0] = self.__cnot_ladder[0].dot(cnots[i, (i+1) % self.__qnum])
        for i in range(1, upper, 2):
            self.__cnot_ladder[0] = self.__cnot_ladder[0].dot(cnots[i, (i+1) % self.__qnum])
            self.__cnot_ladder[1] = self.__cnot_ladder[1].dot(cnots[i, (i+1) % self.__qnum])
        for i in range(0, upper, 2):
            self.__cnot_ladder[1] = self.__cnot_ladder[1].dot(cnots[i, (i+1) % self.__qnum])

    def cnot_ladder(self, stacking):
        '''Applies a one-dimensional ladder of CNOT gates

        Args:
            stacking (int): The even to uneven qubits are coupled first (i.e. 0-1,
                2-3, ... and 1-2, 3-4, ... in the second layer) for stacking=0.
                For stacking=1 the order is reversed.
        '''
        self.vec = self.cnot_ladder[stacking].dot(self.vec)

    def cnot_ladder_lhs(self, stacking):
        warnings.warn('Not implemented.')

    def cnot_ladder_center_matrix(self, stacking):
        warnings.warn('Not implemented.')

    ############################################################################
    # classical Hamiltonian
    def load_classical_ham(self, observable, include_individual_components=False):
        '''Creates a the full vector representing an observable, that is purely classical.

        Args:
            observable (Observable): An observable object.
        '''
        #### NEEDS TO BE REWRITTEN!!!
        observable.check_observable(
            known_keys=['z', 'zz'],
            warning='Non-classical observable component found. Only \'z\' and \'zz\' are accepted in this method.'
        )
        # build gate
        z = np.array([1., -1.])
        id = lambda i: np.full(2**i, 1.)
        self.classical_ham = np.zeros(2**self.__qnum, dtype='double')
        component_list = []
        if 'z' in observable.info:
            for i, weight in enumerate(self.info['z']):
                if weight != None:
                    component = weight * _nkr(id(i), _nkr(z, id(self.__qnum-i-1)))
                    self.classical_ham += component
                    if include_individual_components:
                        component_list.append(component)
        if 'zz' in observable.info:
            for i in range(self.__qnum):
                for j in range(i+1, self.__qnum):
                    if observable.info['zz'][i, j] != None:
                        component = observable.info['zz'][i, j] * _nkr(
                            id(i),
                            _nkr(z, _nkr(id(j-i-1), _nkr(z, id(self.__qnum-j-1))))
                        )
                        self.classical_ham += component
                        if include_individual_components:
                            component_list.append(component)
        if include_individual_components:
            self.classical_ham_components = np.array(component_list)
        return self

    def rot_classical_ham(self, angle):
        '''Rotate around a classical Hamiltonian, as done in QAOA.'''
        self.vec *= np.exp(-1.j * angle * self.classical_ham)

    def rot_classical_ham_lhs(self, angle):
        warnings.warn('Not implemented.')

    def rot_classical_ham_center_matrix(self, angle):
        warnings.warn('Not implemented.')

    def rot_classical_ham_component(self, angle, i):
        '''Rotate around a classical Hamiltonian component, as done in QAOA.'''
        self.vec *= np.exp(-1.j * angle * self.classical_ham_components[i])

    def rot_classical_ham_component_lhs(self, angle, i):
        warnings.warn('Not implemented.')

    def rot_classical_ham_component_center_matrix(self, angle, i):
        warnings.warn('Not implemented.')

    def classical_ham(self):
        '''Multiply a classical Hamiltonian (incl. -1.j) with the state. For derivatives.'''
        self.vec *= -1.j * self.classical_ham

    def classical_ham_lhs(self):
        warnings.warn('Not implemented.')

    def classical_ham_center_matrix(self):
        warnings.warn('Not implemented.')

    ############################################################################
    # checking that the 2-norm is 1.
    def norm_error(self):
        return 1. - np.linalg.norm(self.vec)

    ############################################################################
    # private methods
    def __cnot(self, i, j):
        '''Controlled NOT gate. First argument is the control qubit.'''
        if i < j and j < self.__qnum:
            out1 = _skr(_id(i), _proj0)
            out1 = _skr(out1, _id(self.__qnum-i-1))

            out2 = _skr(_id(i), _proj1)
            out2 = _skr(out2, _id(j-i-1))
            out2 = _skr(out2, _x)
            out2 = _skr(out2, _id(self.__qnum-j-1))
        elif i > j and i < self.__qnum:
            out1 = _skr(_id(i), _proj0)
            out1 = _skr(out1, _id(self.__qnum-i-1))

            out2 = _skr(_id(j), _x)
            out2 = _skr(out2, _id(i-j-1))
            out2 = _skr(out2, _proj1)
            out2 = _skr(out2, _id(self.__qnum-i-1))
        else:
            raise ValueError('Invalid CNOT indecies {} and {}, for {} qubits.'.format(i, j, self.__qnum))
        return (out1 + out2).asformat('csr')


_skr = sp.kron
_nkr = np.kron
_x = sp.coo_matrix([[0., 1.], [1., 0.]], dtype='complex')
_y = sp.coo_matrix([[0., -1.j], [1.j, 0.]], dtype='complex')
_id = lambda i: sp.identity(2**i, dtype='complex', format='coo')
_proj0 = sp.coo_matrix([[1., 0.], [0., 0.]], dtype='complex')
_proj1 = sp.coo_matrix([[0., 0.], [0., 1.]], dtype='complex')
