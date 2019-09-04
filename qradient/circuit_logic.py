from qradient.physical_components import Gates, State, Observable
import numpy as np
import scipy.sparse as sp
from scipy import stats
from tqdm import tqdm_notebook, tnrange
import warnings

def progbar_range(hide_progbar):
    '''Reduces code by returning a regular range or a tnrange.

    A tnrange displays a progress bar while executing.'''
    if hide_progbar:
        return np.arange
    else:
        return tnrange

class ParametrizedCircuit:
    '''Parent class for VQE circuits. Not meant for instantiation.'''
    def init(self, qubit_number, observable, use_observable_components=False):
        self.qnum = qubit_number
        self.state = State(qubit_number) # gates must be added by particular child class
        self.observable = Observable(qubit_number, observable, use_observable_components)
        self.tmp_vec = np.ndarray(2**self.qnum, dtype='complex')
        # FLAGS
        self.has_loaded_projectors = False

    def load_observable(self, observable, use_observable_components):
        '''For loading a new observable'''
        self.observable = Observable(self.qnum, observable, use_observable_components)
        self.has_loaded_projectors = False

    def expec_val(self):
        return self.state.vec.conj().dot(
            self.observable.matrix.dot(self.state.vec)
        ).real

    def sample_expec_val(self, shot_num):
        if not self.has_loaded_projectors:
            self.observable.load_projectors()
            self.has_loaded_projectors = True
        expec_val = 0.
        for i, proj in np.ndenumerate(self.observable.projectors):
            prob = (np.abs(proj.dot(self.state.vec))**2).sum()
            weight = self.observable.projector_weights[i]
            rnd_var = stats.rv_discrete(values=([0, 1], [prob, 1-prob]))
            samples = np.array([weight, -weight])[rnd_var.rvs(size=shot_num)]
            expec_val += samples.mean()
        return expec_val

    def sample_component_expec_val(self, shot_num, component):
        if not self.has_loaded_projectors:
            self.observable.load_projectors()
            self.has_loaded_projectors = True
        expec_val = 0.
        proj = self.observable.projectors[component]
        prob = (np.abs(proj.dot(self.state.vec))**2).sum()
        weight = self.observable.projector_weights[component]
        rnd_var = stats.rv_discrete(values=([0, 1], [prob, 1-prob]))
        samples = np.array([weight, -weight])[rnd_var.rvs(size=shot_num)]
        expec_val += samples.mean()
        return expec_val

    # To do: add here another method which is similar to sample_expec_val, but doesnt loop through all projectors
    # ideally, this method should just sample the projectors corresponding to a particular observable

class McClean(ParametrizedCircuit):
    def __init__(self, qubit_number, observable, layer_number, use_observable_components=False, **kwargs):
        ParametrizedCircuit.init(self, qubit_number, observable, use_observable_components)
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
        self.has_loaded_component_eigensystems = False

    def load_observable(self, observable, use_observable_components=False):
        ParametrizedCircuit.load_observable(self, observable, use_observable_components)
        self.has_loaded_eigensystem = False
        self.has_loaded_component_eigensystems = False

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
        myrange = progbar_range(hide_progbar)
        for i in myrange(self.lnum):
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
        myrange = progbar_range(hide_progbar)
        for i in myrange(self.lnum):
            self.state.cnot_ladder(0)
            self.state_history[i] = self.state.vec
            for q in qrange:
                self.__rot(i, q)
        self.state_history[self.lnum] = self.state.vec
        # calculate expectation value wrt to full observable
        self.state.multiply_matrix(self.observable.matrix)
        expec_val = self.state_history[-1].conj().dot(self.state.vec).real
        # calculate gradient
        for i in myrange(self.lnum-1, -1, -1):
            for q in qrange:
                self.__rot(i, q, angle_sign=-1.)
            self.tmp_vec[:] = self.state.vec
            for q in qrange:
                self.__rot(i, q)
                self.__drot(i, q, angle_sign=-1.)
                grad[i, q] = -2. * self.state_history[i].conj().dot(self.state.vec).real # -1 due to dagger of derivative
                self.state.vec[:] = self.tmp_vec
            self.state.cnot_ladder(1)
        return expec_val, grad

    def grad_run_with_component_sampling(self, hide_progbar=True, ini_state=None):
        if ini_state is None:
            self.state.reset()
        else:
            self.state.vec = ini_state
        qrange = np.arange(self.qnum)
        grad = np.ndarray([self.lnum, self.qnum], dtype='double')
        # run circuit
        for q in qrange:
            self.state.yrot(np.pi/4., q)
        myrange = progbar_range(hide_progbar)
        for i in myrange(self.lnum):
            self.state.cnot_ladder(0)
            self.state_history[i] = self.state.vec
            for q in qrange:
                self.__rot(i, q)
        self.state_history[self.lnum] = self.state.vec
        # calculate expectation value wrt to full observable
        self.state.multiply_matrix(self.observable.matrix)
        expec_val = self.state_history[-1].conj().dot(self.state.vec).real
        # adjust state wrt sampled observable component.
        observable_component = np.random.choice(np.arange(self.observable.num_components), p=self.observable.weight_distribution)
        self.state.vec = self.state_history[-1]
        self.state.multiply_matrix(self.observable.component_array[observable_component])
        # calculate gradient
        for i in myrange(self.lnum-1, -1, -1):
            for q in qrange:
                self.__rot(i, q, angle_sign=-1.)
            self.tmp_vec[:] = self.state.vec
            for q in qrange:
                self.__rot(i, q)
                self.__drot(i, q, angle_sign=-1.)
                grad[i, q] = -2. * self.state_history[i].conj().dot(self.state.vec).real # -1 due to dagger of derivative
                self.state.vec[:] = self.tmp_vec
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
        myrange = progbar_range(hide_progbar)
        for i in myrange(self.lnum):
            self.state.cnot_ladder(0)
            for q in qrange:
                self.__rot(i, q)
            self.state_history[i] = self.state.vec
        # calculate gradient
        if exact_expec_val:
            expec_val = self.expec_val()
        else:
            expec_val = self.sample_expec_val(shot_num)
        # run circuit again with parameter shifts
        for i in myrange(self.lnum):
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

    def sample_grad_with_component_sampling(self, hide_progbar=True, shot_num=1, exact_expec_val=True, ini_state=None):
        if ini_state is None:
            self.state.reset()
        else:
            self.state.vec = ini_state
        qrange = np.arange(self.qnum)
        grad = np.ndarray([self.lnum, self.qnum], dtype='double')
        # run circuit
        for q in qrange:
            self.state.yrot(np.pi/4., q)
        myrange = progbar_range(hide_progbar)
        for i in myrange(self.lnum):
            self.state.cnot_ladder(0)
            for q in qrange:
                self.__rot(i, q)
            self.state_history[i] = self.state.vec
        # calculate gradient
        if exact_expec_val:
            expec_val = self.expec_val()
        else:
            expec_val = self.sample_expec_val(shot_num)
        # run circuit again with parameter shifts
        for i in myrange(self.lnum):
            for dq in qrange:
                observable_component = np.random.choice(np.arange(self.observable.num_components), p=self.observable.weight_distribution)
                self.state.vec = self.state_history[i]
                self.__manual_rot(i, dq, np.pi/2)
                for j in np.arange(i+1, self.lnum):
                    self.state.cnot_ladder(0)
                    for q in qrange:
                        self.__rot(j, q)
                o_plus = self.sample_component_expec_val(shot_num, observable_component)
                self.state.vec = self.state_history[i]
                self.__manual_rot(i, dq, -np.pi/2)
                for j in np.arange(i+1, self.lnum):
                    self.state.cnot_ladder(0)
                    for q in qrange:
                        self.__rot(j, q)
                o_minus = self.sample_component_expec_val(shot_num, observable_component)
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
            self.lhs = McClean.LeftHandSide(eigenvectors.transpose().conj(), self.state.gates)
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
        myrange = progbar_range(hide_progbar)
        for i in myrange(self.lnum):
            self.state.cnot_ladder(0)
            for q in qrange:
                self.__rot(i, q)
            self.state_history[i] = self.state.vec
        # calculate expectation value
        if exact_expec_val:
            expec_val = self.expec_val()
        else:
            expec_val = self.sample_expec_val(shot_num)
        # run reverse circuit
        self.lhs.reset()
        for i in myrange(self.lnum-1): # we don't need the last one
            i_inv = self.lnum - i - 1
            for q in qrange:
                ax, angle = self.axes[i_inv, q], self.angles[i_inv, q]
                self.lhs.rot(ax, angle, q)
            self.lhs.cnot_ladder()
            # Since the lhs matrix is dense after a few layers, we store it dense,
            # but keep it in sparse format for multiplication with sparse gates.
            self.lhs_history[i+1] = self.lhs.matrix.asformat('array')
        # calculate gradient finite-shot measurements
        for i in myrange(self.lnum):
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

    def sample_grad_observable_with_component_sampling(self, shot_num=1, hide_progbar=True, exact_expec_val=True, ini_state=None):
        '''Estimates the gradient by shot_num measurements on a single component of the observable


        Returns the exact expectation value!
        '''
        # calculate eigensystem if not already done. WARNING: DENSE METHOD!!
        if not self.has_loaded_component_eigensystems:

            self.component_eigenvals = []
            self.component_lhs = []
            self.component_lhs_history = []
            for j in range(self.observable.num_components):
                eigenvalues, eigenvectors = np.linalg.eigh(self.observable.component_array[j].asformat('array'))
                lhs = McClean.LeftHandSide(eigenvectors.transpose().conj(), self.state.gates)
                lhs_history = np.ndarray([self.lnum, 2**self.qnum, 2**self.qnum], dtype='complex')
                lhs_history[0] = lhs.ini_matrix.asformat('array')

                self.component_eigenvals.append(eigenvalues)
                self.component_lhs.append(lhs)
                self.component_lhs_history.append(lhs_history)

            self.has_loaded_component_eigensystems = True

        # Sample a component
        observable_component = np.random.choice(np.arange(self.observable.num_components), p=self.observable.weight_distribution)

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
        myrange = progbar_range(hide_progbar)
        for i in myrange(self.lnum):
            self.state.cnot_ladder(0)
            for q in qrange:
                self.__rot(i, q)
            self.state_history[i] = self.state.vec
        # calculate expectation value
        if exact_expec_val:
            expec_val = self.expec_val()
        else:
            expec_val = self.sample_expec_val(shot_num)
        # run reverse circuit
        self.component_lhs[observable_component].reset()
        for i in myrange(self.lnum-1): # we don't need the last one
            i_inv = self.lnum - i - 1
            for q in qrange:
                ax, angle = self.axes[i_inv, q], self.angles[i_inv, q]
                self.component_lhs[observable_component].rot(ax, angle, q)
            self.component_lhs[observable_component].cnot_ladder()
            # Since the lhs matrix is dense after a few layers, we store it dense,
            # but keep it in sparse format for multiplication with sparse gates.
            self.component_lhs_history[observable_component][i+1] = self.component_lhs[observable_component].matrix.asformat('array')
        # calculate gradient finite-shot measurements
        for i in myrange(self.lnum):
            self.state.vec = self.state_history[i]
            for q in qrange:
                self.__manual_rot(i, q, np.pi/2)
                dist = np.abs(self.component_lhs_history[observable_component][self.lnum-i-1].dot(self.state.vec))**2
                rnd_var = stats.rv_discrete(values=(np.arange(2**self.qnum), dist))
                sample1 = (self.component_eigenvals[observable_component][rnd_var.rvs(size=shot_num)]).mean()
                self.__manual_rot(i, q, -np.pi)
                dist = np.abs(self.component_lhs_history[observable_component][self.lnum-i-1].dot(self.state.vec))**2
                rnd_var = stats.rv_discrete(values=(np.arange(2**self.qnum), dist))
                sample2 = (self.component_eigenvals[observable_component][rnd_var.rvs(size=shot_num)]).mean()
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
        myrange = progbar_range(hide_progbar)
        qrange = np.arange(self.qnum)
        for i in myrange(self.lnum):
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
        myrange = progbar_range(hide_progbar)
        for i in myrange(self.dlnum):
            for q in qrange:
                self.state.xrot(data[i, q], q)
                self.state.yrot(encoding_parameters[i, q, 0], q)
                self.state.zrot(encoding_parameters[i, q, 1], q)
            self.state.cnot_ladder(0)
        for i in myrange(self.clnum):
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
        myrange = progbar_range(hide_progbar)
        for i in myrange(self.dlnum):
            for q in qrange:
                self.state.xrot(data[i, q], q)
            self.state_history[2*i] = self.state.vec # save state
            for q in qrange:
                self.state.yrot(encoding_parameters[i, q, 0], q)
            self.state_history[2*i+1] = self.state.vec # save state
            for q in qrange:
                self.state.zrot(encoding_parameters[i, q, 1], q)
            self.state.cnot_ladder(0)
        for i in myrange(self.clnum):
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
        for i in myrange(self.clnum-1, -1, -1):
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
        for i in myrange(self.dlnum-1, -1, -1):
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
        self.state.gates = Gates(qubit_number) \
            .add_xrots() \
            .add_x_summed() \
            .add_classical_ham(self.observable, include_individual_components=True)
            # individul components are used in sample_grad_dense.
        self.lnum = layer_number
        self.state.reset('+') # initialize in uniform-superposition state
        # for calculating the gradient
        self.state_history = np.ndarray([2*self.lnum+1, 2**self.qnum], dtype='complex')
        # FLAGS
        self.has_loaded_eigensystem = False

    def load_observable(self, observable):
        ParametrizedCircuit.load_observable(self, observable)
        self.has_loaded_eigensystem = False
        # Hamiltonian gate components
        self.state.gates.add_classical_ham(self.observable.to_vec())

    def run_expec_val(self, betas, gammas, hide_progbar=True, exact_expec_val=True, shot_num=1, ini_state=None):
        '''Runs the circuit and returns the expectation value under observable'''
        if ini_state is None:
            self.state.reset()
        else:
            self.state.vec = ini_state
        self.__check_parameters(betas, gammas)
        myrange = progbar_range(hide_progbar)
        # run circuit
        for i in myrange(self.lnum):
            self.state.exp_ham_classical(gammas[i])
            self.__xrot_all(betas[i])
        if exact_expec_val:
            return self.expec_val()
        else:
            return self.sample_expec_val(shot_num)

    def grad_run(self, betas, gammas, hide_progbar=True, ini_state=None):
        if ini_state is None:
            self.state.reset()
        else:
            self.state.vec = ini_state
        self.__check_parameters(betas, gammas)
        grad = np.ndarray([self.lnum, 2], dtype='double')
        # run circuit
        myrange = progbar_range(hide_progbar)
        for i in myrange(self.lnum):
            self.state_history[2*i] = self.state.vec
            self.state.exp_ham_classical(gammas[i])
            self.state_history[2*i+1] = self.state.vec
            self.__xrot_all(betas[i])
        self.state_history[2*self.lnum] = self.state.vec
        # calculate expecation value
        self.state.vec *= self.state.gates.classical_ham
        expec_val = self.state_history[2*self.lnum].conj().dot(self.state.vec).real
        # calculate gradient
        for i in myrange(self.lnum-1, -1, -1):
            self.__xrot_all(-betas[i])
            self.tmp_vec[:] = self.state.vec
            self.state.x_summed()
            grad[i, 0] = -2. * self.state_history[2*i+1].conj().dot(self.state.vec).real
            self.state.vec[:] = self.tmp_vec
            self.state.exp_ham_classical(-gammas[i])
            self.tmp_vec[:] = self.state.vec
            self.state.ham_classical()
            grad[i, 1] = -2. * self.state_history[2*i].conj().dot(self.state.vec).real
            self.state.vec[:] = self.tmp_vec
        return expec_val, grad

    def sample_grad(
        self,
        betas,
        gammas,
        shot_num=1,
        hide_progbar=True,
        exact_expec_val=True,
        ini_state=None
    ):
        warnings.warn('Not implemented yet.')

    def sample_grad_dense(
        self,
        betas,
        gammas,
        shot_num=1,
        hide_progbar=True,
        exact_expec_val=True,
        ini_state=None
    ):
        # calculate eigensystem if not already done. WARNING: DENSE METHOD!!
        if not self.has_loaded_eigensystem:
            self.eigenvalues = self.state.gates.classical_ham
            self.lhs = Qaoa.LeftHandSide(
                sp.identity(2**self.qnum, dtype='complex', format='csr'),
                self.state.gates
            )
            self.lhs_history = np.ndarray([2*self.lnum, 2**self.qnum, 2**self.qnum], dtype='complex')
            self.lhs_history[0] = self.lhs.ini_matrix.asformat('array')
            # for collecting samples
            self.samples_plus_comp = np.ndarray(self.state.gates.classical_ham_components.shape[0], dtype='double')
            self.samples_minus_comp = np.ndarray(self.state.gates.classical_ham_components.shape[0], dtype='double')
            self.samples_plus = np.ndarray(self.qnum, dtype='double')
            self.samples_minus = np.ndarray(self.qnum, dtype='double')
            self.has_loaded_eigensystem = True
        qrange = np.arange(self.qnum)
        # prepare to run circuit
        if ini_state is None:
            self.state.reset()
        else:
            self.state.vec = ini_state
        grad = np.ndarray([self.lnum, 2], dtype='double')
        # run circuit
        myrange = progbar_range(hide_progbar)
        for i in myrange(self.lnum):
            self.state.exp_ham_classical(gammas[i])
            self.state_history[2*i] = self.state.vec
            self.__xrot_all(betas[i])
            self.state_history[2*i+1] = self.state.vec
        # calculate expectation value
        if exact_expec_val:
            self.state.vec *= self.state.gates.classical_ham
            expec_val = self.state_history[2*self.lnum-1].conj().dot(self.state.vec).real
        else:
            expec_val = self.sample_expec_val(shot_num)
        # run reverse circuit
        self.lhs.reset()
        for i in myrange(self.lnum):
            i_inv = self.lnum - i - 1
            self.lhs_history[2*i] = self.lhs.matrix.asformat('array')
            self.lhs.xrot_all(betas[i_inv])
            self.lhs_history[2*i+1] = self.lhs.matrix.asformat('array')
            self.lhs.exp_ham_classical(gammas[i_inv])
        # calculate gradient finite-shot measurements
        component_range = np.arange(self.state.gates.classical_ham_components.shape[0])
        for i in myrange(self.lnum):
            # gamma[i]
            for j in component_range:
                self.state.vec[:] = self.state_history[2*i]
                self.state.exp_ham_classical_component(np.pi/4, j)
                dist = np.abs(self.lhs_history[2 * (self.lnum-1-i) + 1].dot(self.state.vec))**2
                self.samples_plus_comp[j] = self.__sample(dist, shot_num)
                self.state.exp_ham_classical_component(-np.pi/2, j)
                dist = np.abs(self.lhs_history[2 * (self.lnum-1-i) + 1].dot(self.state.vec))**2
                self.samples_minus_comp[j] = self.__sample(dist, shot_num)
            grad[i, 1] = (self.samples_plus_comp - self.samples_minus_comp).sum()
            # beta[i]
            for q in qrange:
                self.state.vec[:] = self.state_history[2*i+1]
                self.state.xrot(np.pi/2, q)
                dist = np.abs(self.lhs_history[2 * (self.lnum-1-i)].dot(self.state.vec))**2
                self.samples_plus[q] = self.__sample(dist, shot_num)
                self.state.xrot(-np.pi, q)
                dist = np.abs(self.lhs_history[2 * (self.lnum-1-i)].dot(self.state.vec))**2
                self.samples_minus[q] = self.__sample(dist, shot_num)
            grad[i, 0] = .5 * (self.samples_plus - self.samples_minus).sum()
        return expec_val, grad

    class LeftHandSide:
        '''A helper class for sample_grad_dense.'''
        def __init__(self, matrix, gates):
            # the matrix itself quickly becomes dense, but multiplication with sparse
            # gates is still more efficient with sparse method.
            self.ini_matrix = sp.csr_matrix(matrix, dtype='complex')
            self.matrix = self.ini_matrix
            self.gates = gates
            self.id = sp.identity(2**gates.qnum, dtype='complex', format='csr')
        def exp_ham_classical(self, angle):
            self.matrix = self.matrix.dot(
                sp.diags(
                    np.exp(-1.j * angle * self.gates.classical_ham),
                    format='csr',
                    dtype='complex'
                )
            )
        def xrot_all(self, angle):
            for q in np.arange(self.gates.qnum):
                self.matrix = self.matrix.dot(
                    np.sin(.5*angle) * self.gates.xrot[q] + np.cos(.5*angle) * self.id
                )
        def reset(self):
            self.matrix = self.ini_matrix
    # end LeftHandSide

    def __check_parameters(self, betas, gammas):
        if (betas.size != self.lnum) or (gammas.size != self.lnum):
            raise ValueError((
                'Wrong amount of parameters. Expected {} and {},'.format(self.lnum, self.lnum),
                ' found {} and {}.'.format(len(betas), len(gammas))
            ))
    def __xrot_all(self, angle):
        for q in np.arange(self.qnum):
            self.state.xrot(angle, q)

    def __sample(self, dist, shot_num):
        rnd_var = stats.rv_discrete(values=(np.arange(2**self.qnum), dist))
        return (self.eigenvalues[rnd_var.rvs(size=shot_num)]).mean()
