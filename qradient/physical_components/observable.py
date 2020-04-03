import scipy.sparse as sp
import numpy as np
import warnings


class Observable:
    '''
    Takes a dictionary specifying the observable and builds the matrix.

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

    def __init__(self, observable):
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
                        op = _kr(_kr(_id(2**i), matrix), _id(2**(self.__qnum-i-1)))
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
                        op = _kr(_kr(_id(2**i), _z), _id(2**(j-i-1)))
                        op = _kr(_kr(op, _z), _id(2**(self.__qnum-j-1)))
                        self.matrix += self.dict['zz'][i, j] * op

    def load_components(self):
        '''
        Loads matrices of the individual components of the observable into
        the following attributes:

            components (np.ndarray[scipy.sparse.csr_matrix])
            component_weights (np.ndarray[np.float64])
            component_distribution (np.ndarray[np.float64]):
                Absolute values of component_weights and its 1-norm is
                normalized to one.

        Does nothing if the components are already loaded.
        '''
        if hasattr(self, 'components'):
            pass
        else:
            component_list = []
            component_weights = []
            # single-qubit components
            for identifier, matrix in [('x', _x), ('y', _y), ('z', _z)]:
                if identifier in self.dict:
                    for i, weight in enumerate(self.dict[identifier]):
                        if weight != None:
                            _weight_check(weight, identifier)
                            op = _kr(_kr(_id(2**i), matrix), _id(2**(self.__qnum-i-1)))
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
                            op = _kr(_kr(_id(2**i), _z), _id(2**(j-i-1)))
                            op = _kr(_kr(op, _z), _id(2**(self.__qnum-j-1)))
                            component_list.append(op)
                            component_weights.append(self.dict['zz'][i, j])
            self.components = np.array(component_list)
            self.component_weights = np.array(component_weights)
            weight_normalization = np.sum(np.abs(component_weights))
            self.component_distribution = np.array(np.abs(component_weights))/weight_normalization

    def load_projectors(self):
        '''
        Loads projectors corresponding to the observable components into the
        following attributes:

            projectors (np.ndarray[scipy.sparse.spmatrix])
            projector_weights (np.ndarray[np.float64])
            projector_distribution (np.ndarray[np.float64]):
                The absolute value of projector_weights with its 1-norm
                normalized to one.

        Does nothing if they have been loaded already.
        '''
        self.check_observable(['x', 'y', 'z', 'zz'])
        if hasattr(self, 'projectors'):
            pass
        else:
            # construct pauli projectors
            projectors = []
            projector_weights = []
            for identifier in ['x', 'y', 'z']:
                if identifier in self.dict:
                    for i in range(self.__qnum):
                        if self.dict[identifier][i] != None:
                            projectors.append(_create_projector(identifier, self.__qnum, i))
                            projector_weights.append(self.dict[identifier][i])
            if 'zz' in self.dict:
                for i in range(self.__qnum):
                    for j in range(self.__qnum):
                        if self.dict['zz'][i, j] != None:
                            projectors.append(_create_projector('zz', self.__qnum, i, j))
                            projector_weights.append(self.dict['zz'][i, j])
            self.projectors = np.array(projectors)
            self.projector_weights = np.array(projector_weights)
            weight_normalization = np.sum(np.abs(projector_weights))
            self.projector_distribution = np.array(np.abs(projector_weights))/weight_normalization

    def check_observable(self, known_keys, warning=None):
        '''
        Tests whether observable only contains known keys and throws a warning otherwise.

        Args:
            known_keys(list[str]):
                Keys of components that the caller is able to process.
            warning(str):
                Warning message in case unknown keys are present.
        '''
        for key in list(self.dict.keys()):
            unknown = True
            for k in known_keys:
                if key == k:
                    unknown = False
            if unknown:
                if warning == None:
                    warnings.warn('Unknown element of observable {} will be ignored.'.format(key))
                else:
                    warnings.warn(warning)


def _create_projector(self, key, qnum, *args):
    '''
    Creates a projector matrix.

    Args:
        key (str): The type of projector.
        qnum (int): The system size.
        *args (int): Index or indecies denoting the support of the projector.

    Returns:
        scipy.sparse.spmatrix: The projector, either in 'dia' or 'csr' format.

    Raises:
        ValueError: If the key is not known.
    '''
    if key == 'x':
        i = args[0]
        x_proj_pos = sp.csr_matrix([[.5, .5], [.5, .5]], dtype='complex')  # projects on the +1 subspace
        id1 = _id(2**i, dtype='complex')
        id2 = _id(2**(qnum-i-1), dtype='complex')
        return _kr(id1, _kr(x_proj_pos, id2), format='csr')
    elif key == 'y':
        i = args[0]
        y_proj_pos = sp.csr_matrix([[.5, -.5j], [.5j, .5]], dtype='complex')  # projects on the +1 subspace
        id1 = _id(2**i, dtype='complex')
        id2 = _id(2**(qnum-i-1), dtype='complex')
        return _kr(id1, _kr(y_proj_pos, id2), format='csr')
    elif key == 'z':
        i = args[0]
        z_proj_pos = np.array([1, 0], dtype='complex')  # projects on the +1 subspace
        id1 = np.ones(2**i, dtype='complex')
        id2 = np.ones(2**(qnum-i-1), dtype='complex')
        return sp.diags(np.kron(id1, np.kron(z_proj_pos, id2)), dtype='complex')
    elif key == 'zz':
        i, j = args[0], args[1]
        z_proj_pos = np.array([1, 0], dtype='complex')  # projects on the +1 subspace
        z_proj_neg = np.array([0, 1], dtype='complex')  # projects on the -1 subspace
        id1 = np.ones(2**i, dtype='complex')
        id2 = np.ones(2**(qnum-i-1), dtype='complex')
        id3 = np.ones(2**(j-i-1), dtype='complex')
        id4 = np.ones(2**(qnum-j-1), dtype='complex')
        upup = np.kron(
            id1,
            np.kron(z_proj_pos, np.kron(id3, np.kron(z_proj_pos, id4)))
        )
        downdown = np.kron(
            id1,
            np.kron(z_proj_neg, np.kron(id3, np.kron(z_proj_neg, id4)))
        )
        return sp.diags(upup + downdown, dtype='complex')
    else:
        raise ValueError('Unknown key for projector {}.'.format(key))


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


_kr = sp.kron
_id = lambda i: _id(2**i, dtype='complex', format='coo')
_x = sp.csr_matrix([[0., 1.], [1., 0.]], dtype='complex')
_y = sp.csr_matrix([[0., -1.j], [1.j, 0.]], dtype='complex')
_z = sp.csr_matrix([[1., 0.], [0., -1.]], dtype='complex')
