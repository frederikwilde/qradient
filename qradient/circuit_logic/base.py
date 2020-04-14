from qradient.physical_components import Observable
import numpy as np


class ParametrizedCircuit:
    '''Parent class for VQE circuits. Not meant for instantiation.'''

    def _read_observable(self, observable):
        self._qnum = next(iter(observable.values())).shape[0]
        self.observable = Observable(observable)
        self._tmp_vec = np.ndarray(2**self._qnum, dtype='complex')

def binary_sample(p, shot_num):
    '''
    Samples from a biased coin where the outcome +1 is achieved with probability
    p and the outcome -1 is achieved with probability 1-p.

    Args:
        p (float): Probability of the outcome +1
        shot_num: Number of samples to generate.

    Returns:
        float: The mean of all shot_num samples.
    '''
    return np.sign(np.random.rand(shot_num) - 1. + p).mean()
