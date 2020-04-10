from .base import ParametrizedCircuit, binary_sample
from qradient.physical_components import State
import scipy.sparse as sp
import numpy as np
import warnings


class Qaoa(ParametrizedCircuit):
    def __init__(self, observable, layer_number):
        self._read_observable(observable)
        self.state = State(self._qnum, ini='+')
        self.state.load_xrots()
        self.state.load_allx()
        self._lnum = layer_number
        self._state_history = np.ndarray([2*self._lnum+1, 2**self._qnum], dtype='complex')

    def __run(self, betas, gammas, save_history=False):
        '''
        Only runs the circuit on the current state and saves to
        state_history if specified.
        '''
        if save_history:
            for i in np.arange(self._lnum):
                self._state_history[2*i] = self.state.vec
                self.state.vec = self.observable.exp_dot(gammas[i], self.state.vec)
                self._state_history[2*i+1] = self.state.vec
                self.state.allxrot(betas[i])
            self._state_history[2*self._lnum] = self.state.vec
        else:
            for i in np.arange(self._lnum):
                self.state.vec = self.observable.exp_dot(gammas[i], self.state.vec)
                self.state.allxrot(betas[i])

    def __expec_val(self, expec_val_component, expec_val_shotnum):
        '''
        Calculates the expectation value after __run has been executed with
        active history saving along with multiplication of the state vector with
        the observable.
        This is useful for the gradient methods as one might want a different
        shot number or component to compute the expectation value.
        '''
        if expec_val_component is not None:
            self._tmp_active_component = self.observable.active_component
            self.observable.active_component = expec_val_component
            expec_val = self.observable.expectation_value(
                self._state_history[2*self._lnum],
                shot_num=expec_val_shotnum
            )
            self.observable.active_component = self._tmp_active_component
            return expec_val
        else:
            if expec_val_shotnum == 0:
                return self._state_history[2*self._lnum].conj().dot(self.state.vec).real
            else:
                return self.observable.expectation_value(
                    self._state_history[2*self._lnum],
                    shot_num=expec_val_shotnum
                )

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

    def gradient(self, betas, gammas, expec_val_shotnum=0, expec_val_component=None):
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
            expec_val_component (int or None):
                The component with respect to which the expectation value is
                estimated. Default is None which will use the active_component
                attribute specified in the observable.

        Returns:
            np.float64: The expectation value
            np.ndarray[np.float64]: The exact gradient.
        '''
        self.state.reset()
        self.__check_parameters(betas, gammas)
        grad = np.ndarray([self._lnum, 2], dtype='double')
        # run circuit
        self.__run(betas, gammas, save_history=True)
        # calculate expecation value
        self.state.vec = self.observable.dot_component(self.state.vec)
        expec_val = self.__expec_val(expec_val_component, expec_val_shotnum)
        # calculate gradient
        for i in np.arange(self._lnum-1, -1, -1):
            self.state.allxrot(-betas[i])
            self._tmp_vec[:] = self.state.vec
            self.state.allx()
            grad[i, 0] = -2. * self._state_history[2*i+1].conj().dot(self.state.vec).real
            self.state.vec[:] = self._tmp_vec
            self.state.vec = self.observable.exp_dot(-gammas[i], self.state.vec)
            self._tmp_vec[:] = self.state.vec
            self.state.vec = self.observable.matrix.dot(-1.j * self.state.vec)
            grad[i, 1] = -2. * self._state_history[2*i].conj().dot(self.state.vec).real
            self.state.vec[:] = self._tmp_vec
        return expec_val, grad

    def gradient_ps_rule(
        self,
        betas,
        gammas,
        grad_shot_num=1,
        expec_val_shotnum=0,
        expec_val_component=None
    ):
        self.__check_parameters()
        self.state.reset()
        self.observable.load_projectors()
        self.state.activate_center_matrix()
        grad = np.zeros([self._lnum, 2], dtype='double')
        # run circuit on state
        self.__run(betas, gammas, save_history=True)
        # calculate expecation value
        self.state.vec = self.observable.dot_component(self.state.vec)
        expec_val = self.__expec_val(expec_val_component, expec_val_shotnum)
        # determine the components to measure
        if self.observable.active_component == -1:
            projector_list = self.observable.projectors
            weight_list = self.observable.projector_weights
        else:
            projector_list = self.observable.projectors[self.observable.active_component]
            weight_list = self.observable.projector_weights[self.observable.active_component]
        # run the circuit on center_matrix and calculate the gradient
        for i, p in np.ndenumerate(projector_list):
            self.state.center_matrix = p.copy()
            for j in np.arange(self._lnum-1, -1, -1):
                self.state.vec = self._state_history[2*j+2]  # state after xrot
                deriv = 0.
                for q in np.arange(self._qnum):
                    self.state.xrot(np.pi/2, q)
                    prob_front = self.state.vec.conj().dot(
                        self.state.center_matrix.dot(self.state.vec)
                    )
                    # rotate by -pi/2
                    self.state.xrot(-np.pi, q)
                    prob_back = self.state.vec.conj().dot(
                        self.state.center_matrix.dot(self.state.vec)
                    )
                    deriv += binary_sample(prob_front, grad_shot_num) - \
                        binary_sample(prob_back, grad_shot_num)
                grad[j, 0] += weight_list[i] * deriv
                self.state.allxrot_center_matrix(betas[j])
                self.state.vec = self._state_history[2*j+1]  # state after ham rot
                deriv = 0.
                for k in np.arange(self.observable.component_number):
                    self.state.vec = self.observable.exp_dot_component(np.pi/4, k, self.state.vec)  # see about weights!!!
                    prob_front = self.state.vec.conj().dot(
                        self.state.center_matrix.dot(self.state.vec)
                    )
                    self.state.vec = self.observable.exp_dot_component(-np.pi/2, k, self.state.vec)
                    prob_back = self.state.vec.conj().dot(
                        self.state.center_matrix.dot(self.state.vec)
                    )
                    deriv += binary_sample(prob_front, grad_shot_num) - \
                        binary_sample(prob_back, grad_shot_num)
                grad[j, 1] += weight_list * deriv
                if not j == 0:
                    self.state.center_matrix = self.observable.exp_dot(
                        -gammas[j],
                        self.state.center_matrix,
                        sandwich=True
                    )
                    self.state.clean_center_matrix()
        return expec_val, grad

    def __check_parameters(self, betas, gammas):
        if (betas.size != self._lnum) or (gammas.size != self._lnum):
            raise ValueError((
                'Wrong amount of parameters. Expected {} and {},'.format(self._lnum, self._lnum),
                ' found {} and {}.'.format(len(betas), len(gammas))
            ))
