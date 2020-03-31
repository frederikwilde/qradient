import scipy.sparse as sp
import scipy.sparse.linalg as lg
import numpy as np
import warnings
kr = sp.kron

class Observable:
    def __init__(self, observable):
        '''Takes a dictionary specifying the observable and builds the matrix.

        Args:
            observable (dict): can contain the following fields:
                'x': a 1d-array of prefactors for single-qubit Pauli X. Choose None
                    for no matrix.
                'y': 1d-array for Pauli Y.
                'z': 1d-array for Pauli Z.
                'zz': a 2d-array as upper triangular matrix for two-qubit ZZ terms.
        '''
        self.__qnum = next(iter(observable.values())).shape[0]
        for v in observable.values():
            for s in v.shape:
                if s != self.__qnum:
                    raise ValueError((
                        'Inconsistent shapes in observable dictionary.',
                        'Cannot infer qubit_number.'
                    ))

        # load observable
        self.dict = observable
        self.__check_observable(known_keys=['x', 'y', 'z', 'zz'])
        self.load_matrix(observable)


    def load_matrix(self, observable, store_components=False):
        self.matrix = sp.coo_matrix((2**self.qnum, 2**self.qnum), dtype='complex').asformat('csr')
        x = sp.csr_matrix([[0., 1.], [1., 0.]], dtype='complex')
        y = sp.csr_matrix([[0., -1.j], [1.j, 0.]], dtype='complex')
        z = sp.csr_matrix([[1., 0.], [0., -1.]], dtype='complex')

        if self.store_components:
            component_list = []
            component_weights = []

        # single-qubit components
        for identifier, matrix in [('x', x), ('y', y), ('z', z)]:
            if identifier in observable:
                for i, weight in enumerate(observable[identifier]):
                    if weight != None:
                        Observable.__weight_check(weight, identifier)
                        op = kr(kr(sp.identity(2**i), matrix), sp.identity(2**(self.qnum-i-1)))
                        self.matrix += weight * op
                        if self.store_components:
                            component_list.append(weight * op)
                            component_weights.append(weight)
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
                        if self.store_components:
                            component_list.append(observable['zz'][i, j] * op)
                            component_weights.append(observable['zz'][i, j])
        # store the components and weights as arrays
        if self.store_components:
            self.num_components = len(component_weights)
            self.component_array = np.array(component_list)
            weight_normalization = np.sum(component_weights)
            self.weight_distribution = np.array(component_weights)/weight_normalization


    def load_projectors(self):
        if self.has_loaded_projectors: # in case the caller has not checked
            return None
        self.check_observable(known_keys=['x', 'y', 'z', 'zz'])
        # construct pauli projectors
        projectors = []
        projector_weights = []
        Projector.set_qnum(self.qnum)
        for identifier in ['x', 'y', 'z']:
            if identifier in self.info:
                for i in range(self.qnum):
                    if self.info[identifier][i] != None:
                        projectors.append(Projector(identifier, i))
                        projector_weights.append(self.info[identifier][i])
        if 'zz' in self.info:
            for i in range(self.qnum):
                for j in range(self.qnum):
                    if self.info['zz'][i, j] != None:
                        projectors.append(Projector('zz', i, j))
                        projector_weights.append(self.info['zz'][i, j])
        self.projectors = np.array(projectors)
        self.projector_weights = np.array(projector_weights)
        self.has_loaded_projectors = True

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

class Projector:
    def __init__(self, key, *args):
        if key == 'x':
            self.is_classical = False
            i = args[0]
            x_proj_pos = sp.csr_matrix([[.5, .5], [.5, .5]], dtype='complex') # projects on the +1 subspace
            id1 = sp.identity(2**i, dtype='complex')
            id2 = sp.identity(2**(Projector.qnum-i-1), dtype='complex')
            self.array = kr(id1, kr(x_proj_pos, id2), format='csr')
        elif key == 'y':
            self.is_classical = False
            i = args[0]
            y_proj_pos = sp.csr_matrix([[.5, -.5j], [.5j, .5]], dtype='complex') # projects on the +1 subspace
            id1 = sp.identity(2**i, dtype='complex')
            id2 = sp.identity(2**(Projector.qnum-i-1), dtype='complex')
            self.array = kr(id1, kr(y_proj_pos, id2), format='csr')
        elif key == 'z':
            self.is_classical = True
            i = args[0]
            z_proj_pos = np.array([1, 0], dtype='complex') # projects on the +1 subspace
            id1 = np.ones(2**i, dtype='complex')
            id2 = np.ones(2**(Projector.qnum-i-1), dtype='complex')
            self.array = np.kron(id1, np.kron(z_proj_pos, id2))
        elif key == 'zz':
            self.is_classical = True
            i, j = args[0], args[1]
            z_proj_pos = np.array([1, 0], dtype='complex') # projects on the +1 subspace
            z_proj_neg = np.array([0, 1], dtype='complex') # projects on the -1 subspace
            id1 = np.ones(2**i, dtype='complex')
            id2 = np.ones(2**(Projector.qnum-i-1), dtype='complex')
            id3 = np.ones(2**(j-i-1), dtype='complex')
            id4 = np.ones(2**(Projector.qnum-j-1), dtype='complex')
            upup = np.kron(
                id1,
                np.kron(z_proj_pos, np.kron(id3, np.kron(z_proj_pos, id4)))
            )
            downdown = np.kron(
                id1,
                np.kron(z_proj_neg, np.kron(id3, np.kron(z_proj_neg, id4)))
            )
            self.array = upup + downdown
        else:
            raise ValueError('Unknown key for projector {}.'.format(key))

    def set_qnum(qnum):
        Projector.qnum = qnum

    def dot(self, vec):
        if self.is_classical:
            return self.array * vec
        else:
            return self.array.dot(vec)
