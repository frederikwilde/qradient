from qradient.physical_components import Gates, State
import numpy as np
import scipy.sparse as sp
from scipy import stats
from tqdm import tqdm_notebook, tnrange
import sys
import warnings
kr = sp.kron

class ParametrizedCircuit:
    '''Parent class for VQE circuits. Not meant for instantiation.'''
    def init(self, qubit_number, observable):
        self.qnum = qubit_number
        self.load_observable(observable)

    def load_observable(self, observable):
        '''Takes a dictionary specifying the observable and builds the matrix.

        Args:
            observable (dict): can contain the following fields:
                'x': a 1d-array of prefactors for single-qubit Pauli X. Choose None
                    for no matrix.
                'y': 1d-array for Pauli Y.
                'z': 1d-array for Pauli Z.
                'zz': a 2d-array as upper triangular matrix for two-qubit ZZ terms.
        '''
        self.observable = observable
        self.check_observable(known_keys=['x', 'y', 'z', 'zz'])
        self.observable_mat = sp.coo_matrix((2**self.qnum, 2**self.qnum), dtype='complex').asformat('csr')
        x = sp.csr_matrix([[0., 1.], [1., 0.]], dtype='complex')
        y = sp.csr_matrix([[0., -1.j], [1.j, 0.]], dtype='complex')
        z = sp.csr_matrix([[1., 0.], [0., -1.]], dtype='complex')
        if 'x' in observable:
            for i, weight in enumerate(observable['x']):
                if weight != None:
                    op = kr(kr(sp.identity(2**i), x), sp.identity(2**(self.qnum-i-1)))
                    self.observable_mat += weight * op
        if 'y' in observable:
            for i, weight in enumerate(observable['y']):
                if weight != None:
                    op = kr(kr(sp.identity(2**i), y), sp.identity(2**(self.qnum-i-1)))
                    self.observable_mat += weight * op
        if 'z' in observable:
            for i, weight in enumerate(observable['z']):
                if weight != None:
                    op = kr(kr(sp.identity(2**i), z), sp.identity(2**(self.qnum-i-1)))
                    self.observable_mat += weight * op
        if 'zz' in observable:
            for i in range(self.qnum):
                for j in range(i+1, self.qnum):
                    if observable['zz'][i, j] != None:
                        op = kr(kr(sp.identity(2**i), z), sp.identity(2**(j-i-1)))
                        op = kr(kr(op, z), sp.identity(2**(self.qnum-j-1)))
                        self.observable_mat += observable['zz'][i, j] * op

    def check_observable(self, known_keys):
        '''Tests whether observable only contains known keys and throws a warning otherwise.'''
        for key in list(self.observable.keys()):
            unknown = True
            for k in known_keys:
                if key == k:
                    unknown = False
            if unknown:
                warnings.warn('Unknown element of observable {} will be ignored.'.format(key))

    def expec_val(self):
        return self.state.vec.conj().dot(
            self.observable_mat.dot(self.state.vec)
        ).real

class McClean(ParametrizedCircuit):
    def __init__(self, qubit_number, observable, layer_number, **kwargs):
        ParametrizedCircuit.init(self, qubit_number, observable)
        self.lnum = layer_number
        # collect kwargs
        self.axes = kwargs.get('axes', (3 * np.random.rand(self.lnum, self.qnum)).astype('int'))
        self.angles = kwargs.get('angles', 2*np.pi * np.random.rand(self.lnum, self.qnum))
        # load gates
        gates = Gates(self.qnum) \
            .add_xrots() \
            .add_yrots() \
            .add_zrots() \
            .add_cnot_ladder()
        # build state
        self.state = State(self.qnum, gates)
        # for calculating the gradient
        self.state_history = np.ndarray([self.lnum + 1, 2**self.qnum], dtype='complex')
        self.tmp_vec = np.ndarray(2**self.qnum, dtype='complex')
        # FLAGS
        self.has_loaded_eigensystem = False
        self.has_loaded_projectors = False

    def load_observable(self, observable):
        ParametrizedCircuit.load_observable(self, observable)
        self.has_loaded_eigensystem = False
        self.has_loaded_projectors = False

    def run_expec_val(self, hide_progbar=True, ini_state=None):
        if ini_state is None:
            self.state.reset()
        else:
            self.state.vec = ini_state
        qrange = np.arange(self.qnum)
        # run circuit
        for q in qrange:
            self.state.yrot(np.pi/4., q)
        if hide_progbar:
            rng = np.arange
        else:
            rng = tnrange
        for i in rng(self.lnum):
            self.state.cnot_ladder(0)
            for q in qrange:
                self.__rot(i, q)
        return self.expec_val()

    def grad_run(self, hide_progbar=True, ini_state=None):
        if ini_state is None:
            self.state.reset()
        else:
            self.state.vec = ini_state
        qrange = np.arange(self.qnum)
        grad = np.ndarray([self.lnum, self.qnum], dtype='double')
        # run circuit
        for q in qrange:
            self.state.yrot(np.pi/4., q)
        if hide_progbar:
            rng = np.arange
        else:
            rng = tnrange
        for i in rng(self.lnum):
            self.state.cnot_ladder(0)
            self.state_history[i] = self.state.vec
            for q in qrange:
                self.__rot(i, q)
        self.state_history[self.lnum] = self.state.vec
        # calculate expecation value
        self.state.multiply_matrix(self.observable_mat)
        expec_val = self.state_history[-1].conj().dot(self.state.vec).real
        # calculate gradient
        for i in rng(self.lnum-1, -1, -1):
            for q in qrange:
                self.__rot(i, q, angle_sign=-1.)
            self.tmp_vec = self.state.vec
            for q in qrange:
                self.__rot(i, q)
                self.__drot(i, q, angle_sign=-1.)
                grad[i, q] = -2. * self.state_history[i].conj().dot(self.state.vec).real # -1 due to dagger of derivative
                self.state.vec = self.tmp_vec
            self.state.cnot_ladder(1)
        return expec_val, grad

    def sample_grad(self, hide_progbar=True, shot_num=1, exact_expec_val=True, ini_state=None):
        if not self.has_loaded_projectors:
            self.__load_projectors()
            self.has_loaded_projectors = True
        if ini_state is None:
            self.state.reset()
        else:
            self.state.vec = ini_state
        qrange = np.arange(self.qnum)
        grad = np.ndarray([self.lnum, self.qnum], dtype='double')
        # run circuit
        for q in qrange:
            self.state.yrot(np.pi/4., q)
        if hide_progbar:
            rng = np.arange
        else:
            rng = tnrange
        for i in rng(self.lnum):
            self.state.cnot_ladder(0)
            for q in qrange:
                self.__rot(i, q)
            self.state_history[i] = self.state.vec
        # calculate gradient
        if exact_expec_val:
            self.state.multiply_matrix(self.observable_mat)
            expec_val = self.state_history[self.lnum-1].conj().dot(self.state.vec).real
            self.state.vec = self.state_history[self.lnum-1]
        else:
            expec_val = self.__apply_projectors(shot_num)
        # run circuit again with parameter shifts
        for i in rng(self.lnum):
            for dq in qrange:
                self.state.vec = self.state_history[i]
                self.__manual_rot(i, dq, np.pi/2)
                for j in np.arange(i+1, self.lnum):
                    self.state.cnot_ladder(0)
                    for q in qrange:
                        self.__rot(j, q)
                o_plus = self.__apply_projectors(shot_num)
                self.state.vec = self.state_history[i]
                self.__manual_rot(i, dq, -np.pi/2)
                for j in np.arange(i+1, self.lnum):
                    self.state.cnot_ladder(0)
                    for q in qrange:
                        self.__rot(j, q)
                o_minus = self.__apply_projectors(shot_num)
                grad[i, dq] = .5 * (o_plus - o_minus)
        return expec_val, grad

    def sample_grad_observable(self, shot_num=1, hide_progbar=True, exact_expec_val=True, ini_state=None):
        '''Estimates the gradient by shot_num measurements.

        This method assumes that one can measure the observable as it is. For generic
        Hamiltonians that is most likely not true.

        Returns the exact expectation value!
        '''
        # calculate eigensystem if not already done. WARNING: DENSE METHOD!!
        if not self.has_loaded_eigensystem:
            self.eigensystem = np.linalg.eigh(self.observable_mat.asformat('array'))
            self.lhs = McClean.LeftHandSide(self.eigensystem[1], self.state.gates)
            self.lhs_history = np.ndarray([self.lnum, 2**self.qnum, 2**self.qnum], dtype='complex')
            self.has_loaded_eigensystem = True
        if not self.has_loaded_projectors:
            self.__load_projectors()
            self.has_loaded_projectors = True
        # prepare to run circuit
        if ini_state is None:
            self.state.reset()
        else:
            self.state.vec = ini_state
        qrange = np.arange(self.qnum)
        grad = np.ndarray([self.lnum, self.qnum], dtype='double')
        # run circuit
        for q in qrange:
            self.state.yrot(np.pi/4., q)
        if hide_progbar:
            rng = np.arange
        else:
            rng = tnrange
        for i in rng(self.lnum):
            self.state.cnot_ladder(0)
            for q in qrange:
                self.__rot(i, q)
            self.state_history[i] = self.state.vec
        # calculate expectation value
        self.state.multiply_matrix(self.observable_mat)
        expec_val = self.state_history[self.lnum-1].conj().dot(self.state.vec).real
        if exact_expec_val:
            self.state.multiply_matrix(self.observable_mat)
            expec_val = self.state_history[self.lnum-1].conj().dot(self.state.vec).real
        else:
            expec_val = self.__apply_projectors(shot_num)
        # run reverse circuit
        self.lhs.matrix = sp.csr_matrix(self.eigensystem[1]).transpose()
        self.lhs_history[0] = self.eigensystem[1].transpose()
        for i in rng(self.lnum-1): # we don't need the last one
            i_inv = self.lnum - i - 1
            for q in qrange:
                ax, angle = self.axes[i_inv, q], self.angles[i_inv, q]
                self.lhs.rot(ax, angle, q)
            self.lhs.cnot_ladder()
            # Since the lhs matrix is dense after a few layers, we store it dense,
            # but keep it in sparse format for multiplication with sparse gates.
            self.lhs_history[i+1] = self.lhs.matrix.asformat('array')
        # caculate gradient one-shot measurements
        for i in rng(self.lnum):
            self.state.vec = self.state_history[i]
            for q in qrange:
                self.__manual_rot(i, q, np.pi/2)
                dist = np.abs(self.lhs_history[self.lnum-i-1].dot(self.state.vec))**2
                rnd_var = stats.rv_discrete(values=(np.arange(2**self.qnum), dist))
                sample1 = (self.eigensystem[0][rnd_var.rvs(size=shot_num)]).mean()
                self.__manual_rot(i, q, -np.pi)
                dist = np.abs(self.lhs_history[self.lnum-i-1].dot(self.state.vec))**2
                rnd_var = stats.rv_discrete(values=(np.arange(2**self.qnum), dist))
                sample2 = (self.eigensystem[0][rnd_var.rvs(size=shot_num)]).mean()
                self.state.vec = self.state_history[i]
                grad[i, q] = (sample1 - sample2)/2.
        return expec_val, grad

    class LeftHandSide:
        '''A helper class for sample_grad_observable.'''
        def __init__(self, matrix, gates):
            # the matrix itself quickly becomes dense, but multiplication with sparse
            # gates is still more efficient with sparse method.
            self.matrix = sp.csr_matrix(matrix, dtype='complex')
            self.gates = gates
            self.id = sp.identity(2**gates.qnum, dtype='complex', format='csr')
        def rot(self, axis, angle, q):
            if axis == 0:
                self.matrix = self.matrix.dot(
                    np.sin(.5*angle) * self.gates.xrot[q] + np.cos(.5*angle) * self.id
                )
            elif axis == 1:
                self.matrix = self.matrix.dot(
                    np.sin(.5*angle) * self.gates.yrot[q] + np.cos(.5*angle) * self.id
                )
            elif axis == 2:
                self.matrix = self.matrix.dot(
                    np.exp(-.5j*angle) * sp.diags(self.gates.zrot_pos[q]) + \
                        np.exp(.5j*angle) * sp.diags(self.gates.zrot_neg[q])
                )
            else:
                raise ValueError('Invalid axis {}'.format(axis))
        def cnot_ladder(self):
            self.matrix = self.matrix.dot(self.gates.cnot_ladder[0])
    # end LeftHandSide

    def __apply_projectors(self, shot_num):
        self.tmp_vec = self.state.vec # for resetting
        expec_val = 0.
        for i, op in np.ndenumerate(self.projectors):
            norm = np.linalg.norm(self.state.vec)
            self.state.multiply_matrix(op)
            prob = (np.abs(self.state.vec)**2).sum() / norm
            weight = self.projector_weights[i]
            rnd_var = stats.rv_discrete(values=([weight, -weight], [prob, 1-prob]))
            expec_val += rnd_var.rvs(size=shot_num).mean()
            self.state.vec = self.tmp_vec
        return expec_val

    def __load_projectors(self):
        self.check_observable(known_keys=['x', 'y', 'z', 'zz'])
        # construct pauli projectors
        projectors = []
        projector_weights = []
        x_proj_pos = sp.csr_matrix([[.5, .5], [.5, .5]], dtype='complex') # projects on the +1 subspace
        y_proj_pos = sp.csr_matrix([[.5, -.5j], [.5j, .5]], dtype='complex') # projects on the +1 subspace
        z_proj_pos = sp.csr_matrix([[1., 0.], [0., 0.]], dtype='complex') # projects on the +1 subspace
        z_proj_neg = sp.csr_matrix([[0., 0.], [0., 1.]], dtype='complex') # projects on the -1 subspace
        # read the content of observable
        x = self.observable.get('x', np.full(self.qnum, None))
        y = self.observable.get('y', np.full(self.qnum, None))
        z = self.observable.get('z', np.full(self.qnum, None))
        zz = self.observable.get('zz', np.full([self.qnum, self.qnum], None))
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
                projectors.append(sp.diags(self.state.gates.zrot_pos[i], format='csr'))
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

    def __rot(self, i, q, angle_sign=1.):
        ax = self.axes[i, q]
        if ax == 0:
            self.state.xrot(angle_sign * self.angles[i, q], q)
        elif ax == 1:
            self.state.yrot(angle_sign * self.angles[i, q], q)
        elif ax == 2:
            self.state.zrot(angle_sign * self.angles[i, q], q)
        else:
            raise ValueError('Invalid axis {}'.format(ax))

    def __manual_rot(self, i, q, angle):
        ax = self.axes[i, q]
        if ax == 0:
            self.state.xrot(angle, q)
        elif ax == 1:
            self.state.yrot(angle, q)
        elif ax == 2:
            self.state.zrot(angle, q)
        else:
            raise ValueError('Invalid axis {}'.format(ax))

    def __drot(self, i, q, angle_sign=1.):
        ax = self.axes[i, q]
        if ax == 0:
            self.state.dxrot(angle_sign * self.angles[i, q], q)
        elif ax == 1:
            self.state.dyrot(angle_sign * self.angles[i, q], q)
        elif ax == 2:
            self.state.dzrot(angle_sign * self.angles[i, q], q)
        else:
            raise ValueError('Invalid axis {}'.format(ax))

    ############################################################################
    # outdated methods:
    def grad(self, hide_progbar=True):
        ##### Plain and stupid. Not efficient.
        warnings.warn('This function is very inefficient, use grad_run instead.')
        grad = np.ndarray([self.lnum, self.qnum], dtype='double')
        eps = 10.**-8
        if hide_progbar:
            rng = range
        else:
            rng = tnrange
        qrange = np.arange(self.qnum)
        for i in rng(self.lnum):
            for q in qrange:
                self.angles[i, q] += eps
                self.run()
                e2 = self.expec_val()
                self.angles[i, q] -= 2*eps
                self.run()
                e1 = self.expec_val()
                self.angles[i, q] += eps
                grad[i, q] = np.real(e2 - e1) / (2.*eps)
        return grad

class MeynardClassifier(ParametrizedCircuit):
    def __init__(self, qubit_number, data_layer_number, classification_layer_number):
        observable = {'z': np.full(qubit_number, None)}
        observable['z'][0] = 1. # measure only the first qubit
        ParametrizedCircuit.init(self, qubit_number, observable)
        self.dlnum = data_layer_number
        self.clnum = classification_layer_number
        # load gates
        gates = Gates(self.qnum) \
            .add_xrots() \
            .add_yrots() \
            .add_zrots() \
            .add_cnot_ladder(periodic=False)
        # build state
        self.state = State(self.qnum, gates)
        # for calculating the gradient
        self.state_history = np.ndarray([2*self.dlnum + 3*self.clnum + 1, 2**self.qnum], dtype='complex')
        self.tmp_vec = np.ndarray(2**self.qnum, dtype='complex')

    def run(self, data, encoding_parameters, classification_parameters, hide_progbar=True):
        self.state.reset()
        qrange = np.arange(self.qnum)
        if hide_progbar:
            rng = range
        else:
            rng = tnrange
        for i in rng(self.dlnum):
            for q in qrange:
                self.state.xrot(data[i, q], q)
                self.state.yrot(encoding_parameters[i, q, 0], q)
                self.state.zrot(encoding_parameters[i, q, 1], q)
            self.state.cnot_ladder(0)
        for i in rng(self.clnum):
            for q in qrange:
                self.state.xrot(classification_parameters[i, q, 0], q)
                self.state.yrot(classification_parameters[i, q, 1], q)
                self.state.zrot(classification_parameters[i, q, 2], q)
            self.state.cnot_ladder(0)

    def grad_run(self, data, encoding_parameters, classification_parameters, hide_progbar=True):
        self.state.reset()
        qrange = np.arange(self.qnum)
        encoding_grad = np.ndarray([self.dlnum, self.qnum, 2], dtype='double')
        classification_grad = np.ndarray([self.clnum, self.qnum, 3], dtype='double')
        # run circuit
        if hide_progbar:
            rng = np.arange
        else:
            rng = tnrange
        for i in rng(self.dlnum):
            for q in qrange:
                self.state.xrot(data[i, q], q)
            self.state_history[2*i] = self.state.vec # save state
            for q in qrange:
                self.state.yrot(encoding_parameters[i, q, 0], q)
            self.state_history[2*i+1] = self.state.vec # save state
            for q in qrange:
                self.state.zrot(encoding_parameters[i, q, 1], q)
            self.state.cnot_ladder(0)
        for i in rng(self.clnum):
            self.state_history[3*i + 2*self.dlnum] = self.state.vec # save state
            for q in qrange:
                self.state.xrot(classification_parameters[i, q, 0], q)
            self.state_history[3*i+1 + 2*self.dlnum] = self.state.vec # save state
            for q in qrange:
                self.state.yrot(classification_parameters[i, q, 1], q)
            self.state_history[3*i+2 + 2*self.dlnum] = self.state.vec # save state
            for q in qrange:
                self.state.zrot(classification_parameters[i, q, 2], q)
            self.state.cnot_ladder(0)
        self.state_history[2*self.dlnum + 3*self.clnum] = self.state.vec # save final state
        # calculate expectation value
        self.state.multiply_matrix(self.observable_mat)
        expec_val = self.state_history[-1].conj().dot(self.state.vec).real
        # calculate gradient
        # classifier layer
        for i in rng(self.clnum-1, -1, -1):
            self.state.cnot_ladder(1)
            # Z
            for q in qrange:
                self.state.zrot(-classification_parameters[i, q, 2], q)
            self.tmp_vec = self.state.vec
            for q in qrange:
                self.state.zrot(classification_parameters[i, q, 2], q)
                self.state.dzrot(-classification_parameters[i, q, 2], q)
                classification_grad[i, q, 2] = -2. * self.state_history[3*i+2 + 2*self.dlnum] \
                    .conj().dot(self.state.vec).real
                self.state.vec = self.tmp_vec
            # Y
            for q in qrange:
                self.state.yrot(-classification_parameters[i, q, 1], q)
            self.tmp_vec = self.state.vec
            for q in qrange:
                self.state.yrot(classification_parameters[i, q, 1], q)
                self.state.dyrot(-classification_parameters[i, q, 1], q)
                classification_grad[i, q, 1] = -2. * self.state_history[3*i+1 + 2*self.dlnum] \
                    .conj().dot(self.state.vec).real
                self.state.vec = self.tmp_vec
            # X
            for q in qrange:
                self.state.xrot(-classification_parameters[i, q, 0], q)
            self.tmp_vec = self.state.vec
            for q in qrange:
                self.state.xrot(classification_parameters[i, q, 0], q)
                self.state.dxrot(-classification_parameters[i, q, 0], q)
                classification_grad[i, q, 0] = -2. * self.state_history[3*i + 2*self.dlnum] \
                    .conj().dot(self.state.vec).real
                self.state.vec = self.tmp_vec
        # data encoding layer
        for i in rng(self.dlnum-1, -1, -1):
            self.state.cnot_ladder(1)
            # Z
            for q in qrange:
                self.state.zrot(-encoding_parameters[i, q, 1], q)
            self.tmp_vec = self.state.vec
            for q in qrange:
                self.state.zrot(encoding_parameters[i, q, 1], q)
                self.state.dzrot(-encoding_parameters[i, q, 1], q)
                encoding_grad[i, q, 1] = -2. * self.state_history[2*i+1] \
                    .conj().dot(self.state.vec).real
                self.state.vec = self.tmp_vec
            # Y
            for q in qrange:
                self.state.yrot(-encoding_parameters[i, q, 0], q)
            self.tmp_vec = self.state.vec
            for q in qrange:
                self.state.yrot(encoding_parameters[i, q, 0], q)
                self.state.dyrot(-encoding_parameters[i, q, 0], q)
                encoding_grad[i, q, 0] = -2. * self.state_history[2*i] \
                    .conj().dot(self.state.vec).real
                self.state.vec = self.tmp_vec
            # X
            for q in qrange:
                self.state.xrot(-data[i, q], q)
        return expec_val, encoding_grad, classification_grad
