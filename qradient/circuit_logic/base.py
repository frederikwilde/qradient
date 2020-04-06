from qradient.physical_components import Observable
import numpy as np


class ParametrizedCircuit:
    '''Parent class for VQE circuits. Not meant for instantiation.'''

    def _read_observable(self, observable):
        self._qnum = next(iter(observable.values())).shape[0]
        print(self)
        self.observable = Observable(observable)
        self._tmp_vec = np.ndarray(2**self._qnum, dtype='complex')
