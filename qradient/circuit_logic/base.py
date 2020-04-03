from qradient.physical_components import Observable
import numpy as np


class ParametrizedCircuit:
    '''Parent class for VQE circuits. Not meant for instantiation.'''

    def init(self, observable):
        self.__qnum = next(iter(observable.values())).shape[0]
        self.observable = Observable(observable)
        self.__tmp_vec = np.ndarray(2**self.__qnum, dtype='complex')
