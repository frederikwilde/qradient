import numpy as np
from qradient.circuit_logic import Qaoa
from qradient import optimization_problems
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as pnp
from timeit import timeit
from tqdm import tqdm
from argparse import ArgumentParser

# This script does some simple benchmarking of a QAOA circuit
# for MAXCUT, comparing PennyLane against qradient on the
# the task of computing the gradient using the adjoint rule.
#
# For exact expectation values (i.e., infinite shots) the adjoint
# rule can be used in PennyLane, but qradient exploits a lot of
# structure of the QAOA circuit.
# For finite shots, PennyLane runs each circuit given by the
# parameter-shift rule separately. In contrast, qradient uses a
# method similar to the adjoint method to aggregate the circuit
# iteratively, saving a lot of computation time for moderate numbers
# of qubits (<10).

argparser = ArgumentParser()
argparser.add_argument('--infinite-shots', action='store_true')
argparser.add_argument('--finite-shots', action='store_true', )
argparser.add_argument(
    '--run-benchmark',
    action='store_true',
    help=(
        'Run the benchmarks and store them. If not specified, '
        'an already stored file is loaded (if it exists).'
    )
)


def run_timings(num_qubits, p, shots):
    rng = np.random.default_rng(p)

    # Generate random MAXCUT instance
    num_edges = 10
    all_edges = []
    for i in range(num_qubits):
        for j in range(i+1, num_qubits):
            all_edges.append([i, j])
    indeces = rng.choice(range(len(all_edges)), size=num_edges, replace=False)

    problem = optimization_problems.MaxCut(num_qubits, edge_set=np.array(all_edges)[indeces])
    observable = problem.to_observable()

    # random parameters
    gammas = rng.random(1000)
    betas = rng.random(1000)
    # For PennyLane these need to be autograd (alias qml.numpy)arrays
    # qradient applies x rotations with a factor 1/2
    betas_ag, gammas_ag = map(pnp.array, [.5 * betas, gammas])

    ##########
    # qradient
    ##########

    def time_qradient(p, shots):
        circuit = Qaoa(num_qubits, observable, layer_number=p)

        if shots is None:
            def grad_fun():
                _, grad = circuit.grad_run(betas[:p], gammas[:p])
                return grad
            t = timeit("grad_fun()", number=5, globals=locals())
            return grad_fun(), t/5

        def grad_fun():
            _, grad = circuit.sample_grad_dense(
                betas[:p], gammas[:p], shot_num=shots, exact_expec_val=False
            )
            return grad
        t = timeit("grad_fun()", number=5, globals=locals())
        return grad_fun(), t/5


    ###########
    # PennyLane
    ###########
    dev = qml.device('lightning.qubit', wires=num_qubits,)
    ham = qml.Hamiltonian(
        np.full(len(problem.edge_set), 1.),
        [qml.Z(i) @ qml.Z(j) for i, j in problem.edge_set]
    )
    mixer = qml.qaoa.mixers.x_mixer(dev.wires)


    def circuit(betas, gammas):
        for w in dev.wires:
            qml.Hadamard(wires=w)

        for b, g in zip(betas, gammas):
            qml.qaoa.layers.cost_layer(g, ham)
            qml.qaoa.layers.mixer_layer(b, mixer)

        return qml.expval(ham)


    def time_pennylane(p, shots):
        if shots is None:
            circ = qml.QNode(circuit, dev, diff_method="adjoint")

            def grad_fun():
                d_betas, d_gammas = qml.grad(circ)(betas_ag[:p], gammas_ag[:p])
                return np.array([.5 * d_betas, d_gammas]).T

            t = timeit("grad_fun()", number=5, globals=locals())
            return grad_fun(), t/5

        circ = qml.QNode(circuit, dev, diff_method="parameter-shift")
        circ = qml.set_shots(circ, shots)

        def grad_fun():
            d_betas, d_gammas = qml.grad(circ)(betas_ag[:p], gammas_ag[:p])
            return np.array([.5 * d_betas, d_gammas]).T

        t = timeit("grad_fun()", number=5, globals=locals())
        return grad_fun(), t/5


    # Run timings
    grad, t_qradient = time_qradient(p, shots)
    grad_pl, t_pennylane = time_pennylane(p, shots)

    if shots is None:
        assert np.allclose(grad, grad_pl, atol=1e-10)

    return t_qradient, t_pennylane


def infinite_shots(args):
    p_list = [50, 25, 10]
    linestyles = ['-', '--', ':']
    n_list = [18, 16, 14, 12]
    times_lists = []

    if args.run_benchmark:
        for n in tqdm(n_list):
            times_lists.append([run_timings(n, p, None) for p in p_list])
        times_lists = np.array(times_lists).transpose(1, 0, 2)
        np.save('benchmarks/times_lists_exact.npy', times_lists)
    else:
        try:
            times_lists = np.load('benchmarks/times_lists_exact.npy')
        except FileNotFoundError:
            return

    labels = []
    for p, times, ls in zip(p_list, times_lists, linestyles):
        qr = np.array(times)[:, 0]
        pl = np.array(times)[:, 1]
        plt.plot(n_list, 100 * (pl - qr) / qr, linestyle=ls, marker='.', color='red')
        labels.append(f'{p=}')

        # plt.plot(n_list, qr, linestyle=ls, marker='.', color='red')
        # plt.plot(n_list, pl, linestyle=ls, marker='.', color='blue')
        # labels += [f'qradient {p=}', f'PennyLane {p=}']

    plt.legend(labels=labels)
    plt.semilogy()
    plt.grid(True)
    plt.xlabel('n')
    plt.ylabel('Speedup qradient [%]')
    plt.title('QAOA gradients - qradient vs. PennyLane')
    # plt.show()
    plt.savefig('benchmarks/exact.png')


def finite_shots(args):
    p_list = [50, 25, 10]
    linestyles = ['-', '--', ':', '-.']
    n_list = [10, 8]
    times_lists = []
    SHOTS = 100

    if args.run_benchmark:
        for n in tqdm(n_list):
            times_lists.append([run_timings(n, p, shots=SHOTS) for p in p_list])
        times_lists = np.array(times_lists)
        np.save(f'benchmarks/times_lists_shots{SHOTS}.npy', times_lists)
    else:
        try:
            times_lists = np.load(f'benchmarks/times_lists_shots{SHOTS}.npy')
        except FileNotFoundError:
            return

    labels = []
    for n, times, ls in zip(n_list, times_lists, linestyles):
        qr = np.array(times)[:, 0]
        pl = np.array(times)[:, 1]

        plt.plot(p_list, 100 * (pl - qr) / qr, linestyle=ls, marker='.', color='red')
        labels.append(f'{n=}')

    plt.legend(labels=labels)
    plt.xlabel('p')
    plt.ylabel('Speedup qradient [%]')
    plt.title('QAOA gradients - qradient vs. PennyLane')
    plt.grid(True)
    plt.yticks(np.logspace(0, 4, 25))
    plt.semilogy()
    # plt.show()
    plt.savefig(f'benchmarks/finite-shots{SHOTS}.png')


if __name__ == '__main__':
    args = argparser.parse_args()

    if args.infinite_shots:
        infinite_shots(args)
    if args.finite_shots:
        finite_shots(args)
