from qradient.physical_components import State, Observable
import numpy as np
from scipy import stats
import warnings

class ParametrizedCircuit:
    '''Parent class for VQE circuits. Not meant for instantiation.'''
    def init(self, qubit_number, observable, use_observable_components=False):
        self.qnum = qubit_number
        self.state = State(qubit_number) # gates must be added by particular child class
        self.observable = Observable(qubit_number, observable, store_components=use_observable_components)
        self.tmp_vec = np.ndarray(2**self.qnum, dtype='complex')
        # FLAGS
        self.has_loaded_projectors = False

    def expec_val(self):
        return self.state.vec.conj().dot(
            self.observable.matrix.dot(self.state.vec)
        ).real

    def sample_expec_val(self, shot_num):
        if not self.has_loaded_projectors:
            self.observable.load_projectors()
            self.has_loaded_projectors = True
        expec_val = 0.
        for i, proj in np.ndenumerate(self.observable.projectors):
            prob = (np.abs(proj.dot(self.state.vec))**2).sum()
            weight = self.observable.projector_weights[i]
            rnd_var = stats.rv_discrete(values=([0, 1], [prob, 1-prob]))
            samples = np.array([weight, -weight])[rnd_var.rvs(size=shot_num)]
            expec_val += samples.mean()
        return expec_val

    def sample_component_expec_val(self, shot_num, component):
        if not self.has_loaded_projectors:
            self.observable.load_projectors()
            self.has_loaded_projectors = True
        expec_val = 0.
        proj = self.observable.projectors[component]
        prob = (np.abs(proj.dot(self.state.vec))**2).sum()
        weight = self.observable.projector_weights[component]
        rnd_var = stats.rv_discrete(values=([0, 1], [prob, 1-prob]))
        samples = np.array([weight, -weight])[rnd_var.rvs(size=shot_num)]
        expec_val += samples.mean()
        return expec_val

    # To do: add here another method which is similar to sample_expec_val, but doesnt loop through all projectors
    # ideally, this method should just sample the projectors corresponding to a particular observable
