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

        Attributes:
            dict (dict): The original observable dictionary.
            matrix (scipy.sparse.csr_matrix): The full matrix (in sparse format).
            component_array (np.ndarray[scipy.sparse.csr_matrix]):
                Full matrices corresponding to the individual components.
            component_weights (np.ndarray[np.float64]):
                The weights for each component in component_array.
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
        self.check_observable(known_keys=['x', 'y', 'z', 'zz'])
        self.__load_matrix()

    def __load_matrix(self):
        self.matrix = sp.coo_matrix((2**self.__qnum, 2**self.__qnum), dtype='complex').asformat('csr')
        # single-qubit components
        for identifier, matrix in [('x', _x), ('y', _y), ('z', _z)]:
            if identifier in self.dict:
                for i, weight in enumerate(self.dict[identifier]):
                    if weight != None:
                        _weight_check(weight, identifier)
                        op = kr(kr(_id(2**i), matrix), _id(2**(self.__qnum-i-1)))
                        self.matrix += weight * op
        # two-qubit components
        if 'zz' in self.dict:
            for i in range(self.__qnum):
                for j in range(i+1):
                    if self.dict['zz'][i, j] != None:
                        raise ValueError((
                            'zz of observable should be a upper triangular {} '.format(self.__qnum),
                            'by {} matrix. Diagonal and lower triangle should'.format(self.__qnum),
                            'contain None\'s, not {}.'.format(self.dict['zz'][i, j])
                        ))
                for j in range(i+1, self.__qnum):
                    if self.dict['zz'][i, j] != None:
                        _weight_check(self.dict['zz'][i, j], 'y')
                        op = kr(kr(sp.identity(2**i), _z), _id(2**(j-i-1)))
                        op = kr(kr(op, _z), _id(2**(self.__qnum-j-1)))
                        self.matrix += self.dict['zz'][i, j] * op

    def load_components(self):
        if hasattr(self, 'component_array'):
            return None
        component_list = []
        component_weights = []
        # single-qubit components
        for identifier, matrix in [('x', _x), ('y', _y), ('z', _z)]:
            if identifier in self.dict:
                for i, weight in enumerate(self.dict[identifier]):
                    if weight != None:
                        _weight_check(weight, identifier)
                        op = kr(kr(_id(2**i), matrix), _id(2**(self.__qnum-i-1)))
                        component_list.append(op)
                        component_weights.append(weight)
        # two-qubit components
        if 'zz' in self.dict:
            for i in range(self.__qnum):
                for j in range(i+1):
                    if self.dict['zz'][i, j] != None:
                        raise ValueError((
                            'zz of observable should be a upper triangular {} '.format(self.__qnum),
                            'by {} matrix. Diagonal and lower triangle should'.format(self.__qnum),
                            'contain None\'s, not {}.'.format(self.dict['zz'][i, j])
                        ))
                for j in range(i+1, self.__qnum):
                    if self.dict['zz'][i, j] != None:
                        _weight_check(self.dict['zz'][i, j], 'y')
                        op = kr(kr(sp.identity(2**i), _z), _id(2**(j-i-1)))
                        op = kr(kr(op, _z), _id(2**(self.__qnum-j-1)))
                        component_list.append(op)
                        component_weights.append(self.dict['zz'][i, j])
        self.component_array = np.array(component_list)
        self.weight_array = np.array(component_weights)
        weight_normalization = np.sum(component_weights)
        self.weight_distribution = np.array(component_weights)/weight_normalization

    def load_projectors(self):
        if self.has_loaded_projectors: # in case the caller has not checked
            return None
        self.check_observable(known_keys=['x', 'y', 'z', 'zz'])
        # construct pauli projectors
        projectors = []
        projector_weights = []
        Projector.set_qnum(self.__qnum)
        for identifier in ['x', 'y', 'z']:
            if identifier in self.info:
                for i in range(self.__qnum):
                    if self.info[identifier][i] != None:
                        projectors.append(Projector(identifier, i))
                        projector_weights.append(self.info[identifier][i])
        if 'zz' in self.info:
            for i in range(self.__qnum):
                for j in range(self.__qnum):
                    if self.info['zz'][i, j] != None:
                        projectors.append(Projector('zz', i, j))
                        projector_weights.append(self.info['zz'][i, j])
        self.projectors = np.array(projectors)
        self.projector_weights = np.array(projector_weights)
        self.has_loaded_projectors = True

    def check_observable(self, known_keys, warning=None):
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

def _weight_check(weight, observable_component):
    '''
    Checks that component weights are not numerical zeros.

    Args:
        weight (float): The weight to check
        observable_component (str): The name of the component.
    '''
    if abs(weight) < 10.**-15:
        warnings.warn((
            'Weight in observable {} is zero or almost zero.'.format(observable_component),
            ' If you dont\'t want to include it, set it to None.'
        ))

_id = lambda i: sp.identity(2**i, dtype='complex', format='coo')
_x = sp.csr_matrix([[0., 1.], [1., 0.]], dtype='complex')
_y = sp.csr_matrix([[0., -1.j], [1.j, 0.]], dtype='complex')
_z = sp.csr_matrix([[1., 0.], [0., -1.]], dtype='complex')
