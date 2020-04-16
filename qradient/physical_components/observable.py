import scipy.sparse as sp
from scipy import stats
import numpy as np
import warnings


class Observable:
    '''
    Takes a dictionary specifying the observable and builds the matrix.
    *** Sentence about active_component attribute ***

    Args:
        observable (dict): can contain the following fields:
            'x': a 1d-array of prefactors for single-qubit Pauli X. Choose None
                for no matrix.
            'y': 1d-array for Pauli Y.
            'z': 1d-array for Pauli Z.
            'zz': a 2d-array as upper triangular matrix for two-qubit ZZ terms.

    Attributes:
        dict (dict): The original observable dictionary.
        component_number (int):
            Number of components, i.e. non-zero entries in dict.
        matrix (scipy.sparse.csr_matrix): The full matrix (in sparse format).
        active_component (int):
            Decides whether the expectation_value and dot method use the entire
            observable (if set to -1) or only one component. Default is -1.

        ### attributes below need to be loaded with respective methods ###
        components (np.ndarray[scipy.sparse.csr_matrix]):
            Full matrices corresponding to the individual components.
        component_weights (np.ndarray[np.float64]):
            The weights for each component in component_array.
        component_distribution (np.ndarray[np.float64]):
            The absolute value of component_weights normalized to unit 1-norm.
        projectors (np.ndarray[scipy.sparse.spmatrix]):
            Projectors on the +1 eigenspace of the individual components.
        projector_weights (np.ndarray[np.float64]):
            Weights of the projectors, identical with component_weights.
        projector_distribution (np.ndarray[np.float64]):
            The absolute value of projector_weights normalized to unit 1-norm.
    '''

    def __init__(self, observable):
        self.dict = observable
        self.__qnum = next(iter(observable.values())).shape[0]
        # Check that observable dict contains only valid keys and has consistent shapes.
        self.__check_observable(known_keys=['x', 'y', 'z', 'zz'])
        for v in observable.values():
            for s in v.shape:
                if s != self.__qnum:
                    raise ValueError((
                        'Inconsistent shapes in observable dictionary.',
                        'Cannot infer qubit_number.'
                    ))
        # load observable
        self.__load_matrix()
        # check whether it is classical. Used by exp_dot and sparse methods
        is_classical = True
        for key in list(self.dict.keys()):
            if key not in ['z', 'zz']:
                is_classical = False
        self.__is_classical = is_classical
        if self.__is_classical:
            self.matrix = self.matrix.asformat('dia')
        # active component flag
        self.active_component = -1
        '''
        Indicator which component will be used in expectation_value.
        If set to -1, the entire matrix (or all components) will be used.
        '''

    def __load_matrix(self):
        self.matrix = sp.coo_matrix((2**self.__qnum, 2**self.__qnum), dtype='complex').asformat('csr')
        self.component_number = 0
        # single-qubit components
        for identifier, matrix in [('x', _x), ('y', _y), ('z', _z)]:
            if identifier in self.dict:
                for i, weight in enumerate(self.dict[identifier]):
                    if weight is not None:
                        _weight_check(weight, identifier)
                        op = _kr(_kr(_id(2**i), matrix), _id(2**(self.__qnum-i-1)))
                        self.matrix += weight * op
                        self.component_number += 1
        # two-qubit components
        if 'zz' in self.dict:
            for i in range(self.__qnum):
                for j in range(i+1):
                    if self.dict['zz'][i, j] is not None:
                        raise ValueError((
                            'zz of observable should be a upper triangular {} '.format(self.__qnum),
                            'by {} matrix. Diagonal and lower triangle should'.format(self.__qnum),
                            'contain None\'s, not {}.'.format(self.dict['zz'][i, j])
                        ))
                for j in range(i+1, self.__qnum):
                    if self.dict['zz'][i, j] is not None:
                        _weight_check(self.dict['zz'][i, j], 'zz')
                        op = _kr(_kr(_id(2**i), _z), _id(2**(j-i-1)))
                        op = _kr(_kr(op, _z), _id(2**(self.__qnum-j-1)))
                        self.matrix += self.dict['zz'][i, j] * op
                        self.component_number += 1

    def load_components(self):
        '''
        Loads matrices of the individual components of the observable into
        the following attributes:

            components (np.ndarray[scipy.sparse.csr_matrix])
            component_weights (np.ndarray[np.float64])
            component_distribution (np.ndarray[np.float64]):
                Absolute values of component_weights normalized to unit 1-norm.

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
                        if weight is not None:
                            _weight_check(weight, identifier)
                            op = _kr(_kr(_id(2**i), matrix), _id(2**(self.__qnum-i-1)))
                            component_list.append(op)
                            component_weights.append(weight)
            # two-qubit components
            if 'zz' in self.dict:
                for i in range(self.__qnum):
                    for j in range(i+1):
                        if self.dict['zz'][i, j] is not None:
                            raise ValueError((
                                'zz of observable should be a upper triangular {} '.format(self.__qnum),
                                'by {} matrix. Diagonal and lower triangle should'.format(self.__qnum),
                                'contain None\'s, not {}.'.format(self.dict['zz'][i, j])
                            ))
                    for j in range(i+1, self.__qnum):
                        if self.dict['zz'][i, j] is not None:
                            _weight_check(self.dict['zz'][i, j], 'zz')
                            op = _kr(_kr(_id(2**i), _z), _id(2**(j-i-1)))
                            op = _kr(_kr(op, _z), _id(2**(self.__qnum-j-1)))
                            component_list.append(op)
                            component_weights.append(self.dict['zz'][i, j])
            if self.__is_classical:
                for i, c in enumerate(component_list):
                    component_list[i] = c.asformat('dia')
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
                The absolute value of projector_weights normalized to unit 1-norm.

        Does nothing if they have been loaded already.
        '''
        self.__check_observable(['x', 'y', 'z', 'zz'])
        if hasattr(self, 'projectors'):
            pass
        else:
            # construct pauli projectors
            projectors = []
            projector_weights = []
            for identifier in ['x', 'y', 'z']:
                if identifier in self.dict:
                    for i in range(self.__qnum):
                        if self.dict[identifier][i] is not None:
                            projectors.append(_create_projector(identifier, self.__qnum, i))
                            projector_weights.append(self.dict[identifier][i])
            if 'zz' in self.dict:
                for i in range(self.__qnum):
                    for j in range(self.__qnum):
                        if self.dict['zz'][i, j] is not None:
                            projectors.append(_create_projector('zz', self.__qnum, i, j))
                            projector_weights.append(self.dict['zz'][i, j])
            self.projectors = np.array(projectors)
            self.projector_weights = np.array(projector_weights)
            weight_normalization = np.sum(np.abs(projector_weights))
            self.projector_distribution = np.array(np.abs(projector_weights))/weight_normalization

    def expectation_value(self, vec, shot_num=0):
        '''
        Computes the expectation value of the observable with respect to the
        observable or one of its component, which is decided by the
        active_component attribute.

        Args:
            vec (np.ndarray[np.complex128]): State vector.
            shot_num (int):
                Number of measurements to estimate the expectation value.
                Default is 0, which means infinite shots (i.e.) the exact
                expectation value

        Returns:
            np.float64: The expectation value or the estimation thereof.
        '''
        if shot_num == 0:
            if self.active_component == -1:  # expectation value w.r.t. entire observable
                return vec.conj().dot(self.matrix.dot(vec)).real
            elif self.active_component > -1:  # expectation value w.r.t. sepecific component
                self.load_components()
                return self.component_weights * vec.conj().dot(
                    self.components[self.active_component].dot(vec)
                ).real
            else:
                raise ValueError(
                    'Invalid active_component attribute: {}'.format(self.active_component)
                )
        else:
            self.load_projectors()
            if self.active_component == -1:  # expectation value w.r.t. entire observable
                expec_val = 0.
                for i, proj in np.ndenumerate(self.projectors):
                    prob = (np.abs(proj.dot(vec))**2).sum()
                    weight = self.projector_weights[i]
                    rnd_var = stats.rv_discrete(values=([0, 1], [prob, 1-prob]))
                    samples = np.array([weight, -weight])[rnd_var.rvs(size=shot_num)]
                    expec_val += samples.mean()
                return expec_val
            elif self.active_component > -1:  # expectation value w.r.t. sepecific component
                prob = (np.abs(self.projectors[self.active_component].dot(vec))**2).sum()
                weight = self.projector_weights[self.active_component]
                rnd_var = stats.rv_discrete(values=([0, 1], [prob, 1-prob]))
                samples = np.array([weight, -weight])[rnd_var.rvs(size=shot_num)]
                return samples.mean()
            else:
                raise ValueError(
                    'Invalid active_component attribute: {}'.format(self.active_component)
                )

    def dot_component(self, array):
        '''
        Multiplies the array (vector or matrix) with the active component as
        specified in the active_component attribute (-1 implies all components
        are active).

        Args:
            array (np.ndarray): Input array.

        Returns:
            np.ndarray: Of the same shape as input.
        '''
        if self.active_component == -1:
            return self.matrix.dot(array)
        elif self.active_component > -1:
            self.load_components()
            return self.component_weights * self.components[self.active_component].dot(array)
        else:
            raise ValueError(
                'Invalid active_component attribute: {}'.format(self.active_component)
            )

    def exp_dot(self, angle, array, sandwich=False):
        '''
        Multiplies the full observable matrix with -1.j*angle and
        applies its exponential with array.

        Only implemented for classical
        Observable (otherwise expoentiation is generically expensive).

        Args:
            angle (np.float64): Rotation angle.
            array (np.ndarray): Vector or matrix.
            sandwich (bool):
                Only works if array is 2 dimensional. Default is False. If
                True, exp(-1j*angle*matrix) will be multiplied from the left
                and exp(1j*angle*matrix) will be multiplied from the right.

        Returns:
            np.ndarray: Of same shape as input array.
        '''
        if (not self.__is_classical) or self.matrix.getformat() != 'dia':
            raise AttributeError((
                'exp_dot is only implemented for classical observables.',
                'Ensure that the observable matrix is in DIA format with non-zeros only on the main diagonal.'
            ))
        if array.ndim == 1:
            if sandwich:
                warnings.warn('Ignoring sandwich argument as input array is 1-dimensional.')
            return np.exp(-1.j * angle * self.matrix.data[0]) * array
        elif array.ndim == 2:
            if sandwich:
                self.__tmp_matrix = self.matrix.copy()
                self.__tmp_matrix.data[0] = np.exp(-1.j * angle * self.matrix.data[0])
                self.__tmp_matrix2 = self.matrix.copy()
                self.__tmp_matrix2.data[0] = np.exp(1.j * angle * self.matrix.data[0])
                return self.__tmp_matrix.dot(array).dot(self.__tmp_matrix2)
            else:
                self.__tmp_matrix = self.matrix.copy()
                self.__tmp_matrix.data[0] = np.exp(-1.j * angle * self.matrix.data[0])
                return self.__tmp_matrix.dot(array)
        else:
            raise ValueError((
                'Only 1 and 2 dimensional arrays can be passed.',
                'Dimension was {}'.format(array.ndim)
            ))

    def exp_dot_component(self, angle, component, vec):
        '''
        Multiplies an input vector with the exponential of -1.j*angle*component
        where the component is always the specified one (i.e. the active_component
        attribute is ignored by this method!).

        Args:
            angle (np.float64): The rotation angle
            component (int): A positive integer smaller than component_number
            vec (np.ndarray): The input vector

        Returns:
            np.ndarray: The resulting vector.
        '''
        if vec.ndim != 1:
            raise ValueError('Argument vec has to be 1-dimensional.')
        if not self.__is_classical:
            raise AttributeError('exp_dot_component is only implemented for classical observables.')
        self.load_components()
        return np.exp(-1.j*angle * self.component_weights[component]
            * self.components[component].data[0]) * vec

    def __check_observable(self, known_keys, warning=None):
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


def _create_projector(key, qnum, *args):
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
_id = lambda i: sp.identity(i, dtype='complex', format='coo')
_x = sp.csr_matrix([[0., 1.], [1., 0.]], dtype='complex')
_y = sp.csr_matrix([[0., -1.j], [1.j, 0.]], dtype='complex')
_z = sp.csr_matrix([[1., 0.], [0., -1.]], dtype='complex')
