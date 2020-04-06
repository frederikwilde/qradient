from .base import ParametrizedCircuit
from qradient.physical_components import State
import numpy as np
import scipy.sparse as sp
from scipy import stats
import warnings


class Qaoa(ParametrizedCircuit):
    def __init__(self, observable, layer_number):
        ParametrizedCircuit.init(self, observable)
        self.state = State(self.__qnum, ini='+')
        self.state.load_xrots()
        self.state.load_allx()
        self.state.load_classical_ham(observable)
        self.__lnum = layer_number
        self.__state_history = np.ndarray([2*self.lnum+1, 2**self.qnum], dtype='complex')

    def __allxrot(self, angle):
        for q in np.arange(self.qnum):
            self.state.xrot(angle, q)

    def __run(self, betas, gammas, save_history=False):
        '''
        Only runs the circuit on the current state and saves to
        state_history if specified.
        '''
        if save_history:
            for i in np.arange(self.lnum):
                self.__state_history[2*i] = self.state.vec
                self.state.vec = self.observable.exp_dot(gammas[i], self.state.vec)
                self.__state_history[2*i+1] = self.state.vec
                self.__allxrot(betas[i])
        else:
            for i in np.arange(self.lnum):
                self.state.vec = self.observable.exp_dot(gammas[i], self.state.vec)
                self.state.__allxrot(betas[i])

    def run(self, betas, gammas, shot_num=0):
        '''
        Runs the circuit and returns the expectation value with respect to
        the observable.
        '''
        self.state.reset()
        self.__check_parameters(betas, gammas)
        # run circuit
        self.__run(betas, gammas)
        return self.observable.expectation_value(self.state.vec, shot_num=shot_num)

    def gradient(self, betas, gammas, expec_val_shotnum=0):
        '''
        Computes the exact gradient with respect to observable (and its
        internal active_component). Also returns the expectation value which can
        be exact or stochastic.

        Args:
            betas (np.ndarray[np.float64]): Beta parameters.
            gammas (np.ndarray[np.float64]): Gamma parameters.
            expec_val_shotnum (int):
                Number of shots used to estimate the expectation value. Default
                is 0, which gives the exact expectation value.

        Returns:
            np.float64: The expectation value
            np.ndarray[np.float64]: The exact gradient.
        '''
        self.state.reset()
        self.__check_parameters(betas, gammas)
        grad = np.ndarray([self.lnum, 2], dtype='double')
        # run circuit
        self.__run(betas, gammas, save_history=True)
        self.__state_history[2*self.lnum] = self.state.vec
        # calculate expecation value
        self.state.vec = self.observable.dot(self.state.vec)
        if expec_val_shotnum == 0:
            expec_val = self.__state_history[2*self.lnum].conj().dot(self.state.vec).real
        else:
            expec_val = self.observable.expectation_value(
                self.__state_history[2*self.lnum],
                shot_num=expec_val_shotnum
            )
        # calculate gradient
        for i in np.arange(self.lnum-1, -1, -1):
            self.__xrot_all(-betas[i])
            self.__tmp_vec[:] = self.state.vec
            self.state.allx()
            grad[i, 0] = -2. * self.__state_history[2*i+1].conj().dot(self.state.vec).real
            self.state.vec[:] = self.__tmp_vec
            self.state.vec = self.observable.exp_dot(-gammas[i], self.state.vec)
            self.__tmp_vec[:] = self.state.vec
            self.state.vec = self.observable.dot(-1.j * self.state.vec)
            grad[i, 1] = -2. * self.__state_history[2*i].conj().dot(self.state.vec).real
            self.state.vec[:] = self.__tmp_vec
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
        for i in np.arange(self.lnum):
            self.state.exp_ham_classical(gammas[i])
            self.__state_history[2*i] = self.state.vec
            self.__xrot_all(betas[i])
            self.__state_history[2*i+1] = self.state.vec
        # calculate expectation value
        if exact_expec_val:
            self.state.vec *= self.state.gates.classical_ham
            expec_val = self.__state_history[2*self.lnum-1].conj().dot(self.state.vec).real
        else:
            expec_val = self.sample_expec_val(shot_num)
        # run reverse circuit
        self.lhs.reset()
        for i in np.arange(self.lnum):
            i_inv = self.lnum - i - 1
            self.lhs_history[2*i] = self.lhs.matrix.asformat('array')
            self.lhs.xrot_all(betas[i_inv])
            self.lhs_history[2*i+1] = self.lhs.matrix.asformat('array')
            self.lhs.exp_ham_classical(gammas[i_inv])
        # calculate gradient finite-shot measurements
        component_range = np.arange(self.state.gates.classical_ham_components.shape[0])
        for i in np.arange(self.lnum):
            # gamma[i]
            for j in component_range:
                self.state.vec[:] = self.__state_history[2*i]
                self.state.exp_ham_classical_component(np.pi/4, j)
                dist = np.abs(self.lhs_history[2 * (self.lnum-1-i) + 1].dot(self.state.vec))**2
                self.samples_plus_comp[j] = self.__sample(dist, shot_num)
                self.state.exp_ham_classical_component(-np.pi/2, j)
                dist = np.abs(self.lhs_history[2 * (self.lnum-1-i) + 1].dot(self.state.vec))**2
                self.samples_minus_comp[j] = self.__sample(dist, shot_num)
            grad[i, 1] = (self.samples_plus_comp - self.samples_minus_comp).sum()
            # beta[i]
            for q in qrange:
                self.state.vec[:] = self.__state_history[2*i+1]
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

    def __sample(self, dist, shot_num):
        rnd_var = stats.rv_discrete(values=(np.arange(2**self.qnum), dist))
        return (self.eigenvalues[rnd_var.rvs(size=shot_num)]).mean()
