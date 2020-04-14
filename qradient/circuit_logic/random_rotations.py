from .base import ParametrizedCircuit
from qradient.physical_components import State
import numpy as np
import scipy.sparse as sp
from scipy import stats
import warnings

class RandomRotations(ParametrizedCircuit):
    def __init__(self, observable, axes):
        self._read_observable(observable)
        self._lnum = axes.shape[0]
        # collect kwargs
        self.axes = axes
        self.state = State(self._qnum)
        self.state.load_xrots()
        self.state.load_yrots()
        self.state.load_zrots()
        self.state.load_cnot_ladder()
        self._state_history = np.ndarray([self._lnum + 1, 2**self._qnum], dtype='complex')

    def __rot(self, i, q, angle):
        ax = self.axes[i, q]
        if ax == 0:
            self.state.xrot(angle, q)
        elif ax == 1:
            self.state.yrot(angle, q)
        elif ax == 2:
            self.state.zrot(angle, q)
        else:
            raise ValueError('Invalid axis {}'.format(ax))

    def __drot(self, i, q, angle):
        ax = self.axes[i, q]
        if ax == 0:
            self.state.dxrot(angle, q)
        elif ax == 1:
            self.state.dyrot(angle, q)
        elif ax == 2:
            self.state.dzrot(angle, q)
        else:
            raise ValueError('Invalid axis {}'.format(ax))

    def __run(self, angles, save_history=False):
        '''
        Only runs the circuit on the current state and saves to
        state_history if specified.
        '''
        if save_history:
            for q in np.arange(self._qnum):
                self.state.yrot(np.pi/4., q)
            for i in np.arange(self._lnum):
                self.state.cnot_ladder(0)
                self._state_history[i] = self.state.vec
                for q in np.arange(self._qnum):
                    self.__rot(i, q, angles[i, q])
            self._state_history[self._lnum] = self.state.vec
        else:
            for q in np.arange(self._qnum):
                self.state.yrot(np.pi/4., q)
            for i in np.arange(self._lnum):
                self.state.cnot_ladder(0)
                for q in np.arange(self._qnum):
                    self.__rot(i, q, angles[i, q])

    def run(self, angles, shot_num=0):
        '''Runs the circuit and returns the expectation value under observable'''
        self.state.reset()
        # run circuit
        self.__run(angles)
        return self.observable.expectation_value(self.state.vec, shot_num=shot_num)

    def gradient(self, angles, expec_val_shotnum=0, expec_val_component=None):
        self.state.reset()
        grad = np.ndarray([self._lnum, self._qnum], dtype='double')
        # run circuit
        self.__run(angles, save_history=True)
        # calculate expectation value wrt to observable
        self.state.vec = self.observable.matrix.dot(self.state.vec)
        if expec_val_component is not None:
            self._tmp_active_component = self.observable.active_component
            self.observable.active_component = expec_val_component
            expec_val = self.observable.expectation_value(
                self._state_history[-1],
                shot_num=expec_val_shotnum
            )
            self.observable.active_component = self._tmp_active_component
        else:
            if expec_val_shotnum == 0:
                expec_val = self._state_history[-1].conj().dot(self.state.vec).real
            else:
                expec_val = self.observable.expectation_value(
                    self._state_history[-1],
                    shot_num=expec_val_shotnum
                )
        # calculate gradient
        for i in np.arange(self._lnum-1, -1, -1):
            for q in np.arange(self._qnum):
                self.__rot(i, q, -angles[i, q])
            self._tmp_vec[:] = self.state.vec
            for q in np.arange(self._qnum):
                self.__rot(i, q, angles[i, q])
                self.__drot(i, q, -angles[i, q])
                grad[i, q] = -2. * self._state_history[i].conj().dot(self.state.vec).real # -1 due to dagger of derivative
                self.state.vec[:] = self._tmp_vec
            self.state.cnot_ladder(1)
        return expec_val, grad

    def gradient_sample(self, angles, grad_shot_num=1, expec_val_shotnum=0, expec_val_component=None):
        warnings.warn('Not implemented yet.')
