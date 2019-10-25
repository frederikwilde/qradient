from .base import ParametrizedCircuit, progbar_range
from qradient.physical_components import Gates, State, Observable
import numpy as np
import scipy.sparse as sp
from scipy import stats
import warnings

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
        ParametrizedCircuit.load_observable(self, observable, use_observable_components=False)
        self.has_loaded_eigensystem = False
        # Hamiltonian gate components
        self.state.gates.add_classical_ham(self.observable, include_individual_components=True)

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
