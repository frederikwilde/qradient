import numpy as np
from qradient.circuit_logic import Qaoa
from qradient.optimization_problems import MaxCut
from time import time

def f():
    qnum, lnum = 20, 10
    edge_num = 30
    problem = MaxCut(qnum, edge_num).random()
    circuit = Qaoa(qnum, problem.to_observable(), lnum)

    gate = np.zeros(1048576, dtype='complex')
    for i in range(len(circuit.state.gates.classical_ham_weights)):
        gate += circuit.state.gates.classical_ham_weights[i] * circuit.state.gates.classical_ham_components[i]

    t1 = time()
    np.exp(gate)
    t2 = time()
    print(t2-t1)

    t1 = time()
    gate = np.zeros(1048576, dtype='complex')
    for i in range(len(circuit.state.gates.classical_ham_weights)):
        gate += np.exp(circuit.state.gates.classical_ham_weights[i]) * circuit.state.gates.classical_ham_components[i]
    t2 = time()
    print(t2-t1)

    gate = np.zeros(1048576, dtype='complex')
    t1 = time()
    for i in range(len(circuit.state.gates.classical_ham_weights)):
        gate += np.exp(circuit.state.gates.classical_ham_weights[i]) * circuit.state.gates.classical_ham_components[i]
    t2 = time()
    print(t2-t1)
