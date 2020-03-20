from .base import ParametrizedCircuit, progbar_range
from qradient.physical_components import Gates, State, Observable
import numpy as np
import scipy.sparse as sp
from scipy import stats
import warnings

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

    def sample_grad_observable(self, *args):
        warnings.warn(
            'Method sample_grad_observable is now called sample_grad_dense.',
            DeprecationWarning,
            stacklevel=2
        )

    def sample_grad_dense(self, shot_num=1, hide_progbar=True, exact_expec_val=True, ini_state=None):
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

    def sample_grad_observable_with_component_sampling(self, *args):
        warnings.warn(
            'Method sample_grad_observable_with_component_sampling is now called sample_grad_dense_with_component_sampling.',
            DeprecationWarning,
            stacklevel=2
        )

    def sample_grad_dense_with_component_sampling(self, shot_num=1, hide_progbar=True, exact_expec_val=True, ini_state=None):
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
