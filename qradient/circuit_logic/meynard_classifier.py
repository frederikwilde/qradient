from .utils import ParametrizedCircuit, progbar_range
from qradient.physical_components import Gates, State, Observable
import numpy as np

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
