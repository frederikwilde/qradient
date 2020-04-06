from .base import ParametrizedCircuit
from qradient.physical_components import State
import numpy as np
import scipy.sparse as sp
from scipy import stats
import warnings


class Qaoa(ParametrizedCircuit):
    def __init__(self, observable, layer_number):
        self._read_observable(observable)
        self.state = State(self._qnum, ini='+')
        self.state.load_xrots()
        self.state.load_allx()
        self._lnum = layer_number
        self._state_history = np.ndarray([2*self._lnum+1, 2**self._qnum], dtype='complex')

    def __allxrot(self, angle):
        for q in np.arange(self._qnum):
            self.state.xrot(angle, q)

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
                self.__allxrot(betas[i])
        else:
            for i in np.arange(self._lnum):
                self.state.vec = self.observable.exp_dot(gammas[i], self.state.vec)
                self.__allxrot(betas[i])

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
        self._state_history[2*self._lnum] = self.state.vec
        # calculate expecation value
        self.state.vec = self.observable.dot(self.state.vec)
        if expec_val_component is not None:
            self._tmp_active_component = self.observable.active_component
            self.observable.active_component = expec_val_shotnum
            expec_val = self.observable.expectation_value(
                self._state_history[2*self._lnum],
                shot_num=expec_val_shotnum
            )
            self.observable.active_component = self._tmp_active_component
        else:
            if expec_val_shotnum == 0:
                expec_val = self._state_history[2*self._lnum].conj().dot(self.state.vec).real
            else:
                expec_val = self.observable.expectation_value(
                    self._state_history[2*self._lnum],
                    shot_num=expec_val_shotnum
                )
        # calculate gradient
        for i in np.arange(self._lnum-1, -1, -1):
            self.__allxrot(-betas[i])
            self._tmp_vec[:] = self.state.vec
            self.state.allx()
            grad[i, 0] = -2. * self._state_history[2*i+1].conj().dot(self.state.vec).real
            self.state.vec[:] = self._tmp_vec
            self.state.vec = self.observable.exp_dot(-gammas[i], self.state.vec)
            self._tmp_vec[:] = self.state.vec
            self.state.vec = self.observable.dot(-1.j * self.state.vec)
            grad[i, 1] = -2. * self._state_history[2*i].conj().dot(self.state.vec).real
            self.state.vec[:] = self._tmp_vec
        return expec_val, grad

    def gradient_sample(self, betas, gammas, grad_shot_num=1, expec_val_shotnum=0):
        warnings.warn('Not implemented yet.')

    def __check_parameters(self, betas, gammas):
        if (betas.size != self._lnum) or (gammas.size != self._lnum):
            raise ValueError((
                'Wrong amount of parameters. Expected {} and {},'.format(self._lnum, self._lnum),
                ' found {} and {}.'.format(len(betas), len(gammas))
            ))
