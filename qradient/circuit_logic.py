from qradient.physical_components import Gates, State, Observable
import numpy as np
import scipy.sparse as sp
from scipy import stats
from tqdm import tqdm_notebook, tnrange
import warnings

class ParametrizedCircuit:
    '''Parent class for VQE circuits. Not meant for instantiation.'''
    def init(self, qubit_number, observable):
        self.qnum = qubit_number
        self.state = State(qubit_number) # gates must be added by particular child class
        self.observable = Observable(qubit_number, observable)
        self.tmp_vec = np.ndarray(2**self.qnum, dtype='complex')
        # FLAGS
        self.has_loaded_projectors = False

    def load_observable(self, observable):
        '''For loading a new observable'''
        self.observable = Observable(self.qnum, observable)
        self.has_loaded_projectors = False

    def expec_val(self):
        return self.state.vec.conj().dot(
            self.observable.matrix.dot(self.state.vec)
        ).real

    def sample_expec_val(self, shot_num):
        if not self.has_loaded_projectors:
            self.observable.load_projectors()
            self.has_loaded_projectors = True
        self.tmp_vec = self.state.vec # for resetting
        expec_val = 0.
        for i, op in np.ndenumerate(self.observable.projectors):
            self.state.multiply_matrix(op)
            prob = (np.abs(self.state.vec)**2).sum()
            weight = self.observable.projector_weights[i]
            rnd_var = stats.rv_discrete(values=([weight, -weight], [prob, 1-prob]))
            expec_val += rnd_var.rvs(size=shot_num).mean()
            self.state.vec = self.tmp_vec
        return expec_val

class McClean(ParametrizedCircuit):
    def __init__(self, qubit_number, observable, layer_number, **kwargs):
        ParametrizedCircuit.init(self, qubit_number, observable)
        self.lnum = layer_number
        # collect kwargs
        self.axes = kwargs.get('axes', (3 * np.random.rand(self.lnum, self.qnum)).astype('int'))
        self.angles = kwargs.get('angles', 2*np.pi * np.random.rand(self.lnum, self.qnum))
        # load gates
        self.state.gates = Gates(self.qnum) \
            .add_xrots() \
            .add_yrots() \
            .add_zrots() \
            .add_cnot_ladder()
        # for calculating the gradient
        self.state_history = np.ndarray([self.lnum + 1, 2**self.qnum], dtype='complex')
        # FLAGS
        self.has_loaded_eigensystem = False

    def load_observable(self, observable):
        ParametrizedCircuit.load_observable(self, observable)
        self.has_loaded_eigensystem = False

    def run_expec_val(self, hide_progbar=True, exact_expec_val=True, shot_num=1, ini_state=None):
        '''Runs the circuit and returns the expectation value under observable'''
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
        if exact_expec_val:
            return self.expec_val()
        else:
            return self.sample_expec_val(shot_num)

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
        self.state.multiply_matrix(self.observable.matrix)
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
            self.state.multiply_matrix(self.observable.matrix)
            expec_val = self.state_history[self.lnum-1].conj().dot(self.state.vec).real
            self.state.vec = self.state_history[self.lnum-1]
        else:
            expec_val = self.sample_expec_val(shot_num)
        # run circuit again with parameter shifts
        for i in rng(self.lnum):
            for dq in qrange:
                self.state.vec = self.state_history[i]
                self.__manual_rot(i, dq, np.pi/2)
                for j in np.arange(i+1, self.lnum):
                    self.state.cnot_ladder(0)
                    for q in qrange:
                        self.__rot(j, q)
                o_plus = self.sample_expec_val(shot_num)
                self.state.vec = self.state_history[i]
                self.__manual_rot(i, dq, -np.pi/2)
                for j in np.arange(i+1, self.lnum):
                    self.state.cnot_ladder(0)
                    for q in qrange:
                        self.__rot(j, q)
                o_minus = self.sample_expec_val(shot_num)
                grad[i, dq] = .5 * (o_plus - o_minus)
        return expec_val, grad

    # NOTE: rename this to sample_grad_dense
    def sample_grad_observable(self, shot_num=1, hide_progbar=True, exact_expec_val=True, ini_state=None):
        '''Estimates the gradient by shot_num measurements.

        This method assumes that one can measure the observable as it is. For generic
        Hamiltonians that is most likely not true.

        Returns the exact expectation value!
        '''
        # calculate eigensystem if not already done. WARNING: DENSE METHOD!!
        if not self.has_loaded_eigensystem:
            self.eigenvalues, eigenvectors = np.linalg.eigh(self.observable.matrix.asformat('array'))
            self.lhs = McClean.LeftHandSide(eigenvectors.transpose(), self.state.gates)
            self.lhs_history = np.ndarray([self.lnum, 2**self.qnum, 2**self.qnum], dtype='complex')
            self.lhs_history[0] = self.lhs.ini_matrix.asformat('array')
            self.has_loaded_eigensystem = True
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
        if exact_expec_val:
            self.state.multiply_matrix(self.observable.matrix)
            expec_val = self.state_history[self.lnum-1].conj().dot(self.state.vec).real
        else:
            expec_val = self.sample_expec_val(shot_num)
        # run reverse circuit
        self.lhs.reset()
        for i in rng(self.lnum-1): # we don't need the last one
            i_inv = self.lnum - i - 1
            for q in qrange:
                ax, angle = self.axes[i_inv, q], self.angles[i_inv, q]
                self.lhs.rot(ax, angle, q)
            self.lhs.cnot_ladder()
            # Since the lhs matrix is dense after a few layers, we store it dense,
            # but keep it in sparse format for multiplication with sparse gates.
            self.lhs_history[i+1] = self.lhs.matrix.asformat('array')
        # calculate gradient finite-shot measurements
        for i in rng(self.lnum):
            self.state.vec = self.state_history[i]
            for q in qrange:
                self.__manual_rot(i, q, np.pi/2)
                dist = np.abs(self.lhs_history[self.lnum-i-1].dot(self.state.vec))**2
                rnd_var = stats.rv_discrete(values=(np.arange(2**self.qnum), dist))
                sample1 = (self.eigenvalues[rnd_var.rvs(size=shot_num)]).mean()
                self.__manual_rot(i, q, -np.pi)
                dist = np.abs(self.lhs_history[self.lnum-i-1].dot(self.state.vec))**2
                rnd_var = stats.rv_discrete(values=(np.arange(2**self.qnum), dist))
                sample2 = (self.eigenvalues[rnd_var.rvs(size=shot_num)]).mean()
                self.state.vec = self.state_history[i]
                grad[i, q] = (sample1 - sample2)/2.
        return expec_val, grad

    class LeftHandSide:
        '''A helper class for sample_grad_observable.'''
        def __init__(self, matrix, gates):
            # the matrix itself quickly becomes dense, but multiplication with sparse
            # gates is still more efficient with sparse method.
            self.ini_matrix = sp.csr_matrix(matrix, dtype='complex')
            self.matrix = self.ini_matrix
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
        def reset(self):
            self.matrix = self.ini_matrix
    # end LeftHandSide

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
    # outdated method:
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
                e2 = self.run_expec_val()
                self.angles[i, q] -= 2*eps
                e1 = self.run_expec_val()
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
        self.state.gates = Gates(self.qnum) \
            .add_xrots() \
            .add_yrots() \
            .add_zrots() \
            .add_cnot_ladder(periodic=False)
        # for calculating the gradient
        self.state_history = np.ndarray([2*self.dlnum + 3*self.clnum + 1, 2**self.qnum], dtype='complex')

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
        self.state.multiply_matrix(self.observable.matrix)
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

class Qaoa(ParametrizedCircuit):
    def __init__(self, qubit_number, observable, layer_number):
        ParametrizedCircuit.init(self, qubit_number, observable)
        self.lnum = layer_number
        self.state.reset('+') # initialize in uniform-superposition state
        # for calculating the gradient
        self.state_history = np.ndarray([self.lnum + 1, 2**self.qnum], dtype='complex')
        # FLAGS
        self.has_loaded_eigensystem = False

    def load_observable(self, observable):
        ParametrizedCircuit.load_observable(self, observable)
        self.has_loaded_eigensystem = False
        #### load new Hamiltonian gate

    def grad_run(self, betas, gammas, hide_progbar=True):
        pass

    def sample_grad(self, betas, gammas, hide_progbar=True):
        pass

    def sample_grad_dense(self, betas, gammas, hide_progbar=True):
        pass
