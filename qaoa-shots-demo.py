import numpy as np
from qradient.circuit_logic import Qaoa
from qradient import optimization_problems
import matplotlib.pyplot as plt


num_qubits, p = 10, 20

# Generate random MAXCUT instance
problem = optimization_problems.MaxCut(num_qubits, edge_num=10)
observable = problem.to_observable()

## Plot the MAXCUT graph
# plt.figure(figsize=(2,2))
# v, e = problem.plot_lists()
# plt.plot(*(v + e))
# plt.savefig('maxcut-problem.pdf')

circuit = Qaoa(num_qubits, observable, layer_number=p)
rng = np.random.default_rng(42)
gammas = rng.random(p)
betas = rng.random(p)

expec_val, grad = circuit.sample_grad_dense(
    betas, gammas, shot_num=1000, exact_expec_val=False)

print(grad)