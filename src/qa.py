from qiskit import *
from qiskit.circuit import Parameter
from qiskit.circuit.controlflow import for_loop
from qiskit.quantum_info import Operator
from qiskit.circuit.library import PhaseGate
from qiskit_aer import *
import numpy as np
from scipy.linalg import polar
from argparse import *
import time

from . import preprocess
from . import basic


# Construct the quantum circuit for quantum annealing (QA)
# A, B: adjacent matrices
# T0: total evolution time
# M0: number of layers
def QA_circuit(problem, params):
    mat_A = problem.mat_A
    mat_B = problem.mat_B
    vec_A = problem.vec_A
    vec_B = problem.vec_B
    N = problem.N
    M = problem.M
    L = problem.L

    dt = params["t0"] / params["m0"]

    def spin_one(phi, n):
        this = QuantumCircuit(L)
        bin_n = bin(n)[2:]
        for m in range(L):
            if len(bin_n) <= m or bin_n[-m - 1] == "0":
                this.x(L - 1 - m)
        if L > 1:
            this.compose(PhaseGate(phi).control(L - 1), range(L), inplace=True)
        else:
            this.compose(PhaseGate(phi), range(L), inplace=True)
        for m in range(L):
            if len(bin_n) <= m or bin_n[-m - 1] == "0":
                this.x(L - 1 - m)
        return this

    def spin_two(phi, n, m):
        this = QuantumCircuit(2 * L)
        bin_n = bin(n)[2:]
        for mu in range(L):
            if len(bin_n) <= mu or bin_n[-mu - 1] == "0":
                this.x(L - 1 - mu)
        bin_m = bin(m)[2:]
        for mu in range(L):
            if len(bin_m) <= mu or bin_m[-mu - 1] == "0":
                this.x(2 * L - 1 - mu)
        this.compose(PhaseGate(phi).control(2 * L - 1), range(2 * L), inplace=True)
        for mu in range(L):
            if len(bin_n) <= mu or bin_n[-mu - 1] == "0":
                this.x(L - 1 - mu)
        for mu in range(L):
            if len(bin_m) <= mu or bin_m[-mu - 1] == "0":
                this.x(2 * L - 1 - mu)
        return this

    def circuit_unit():
        circ = QuantumCircuit(N * L)
        x = Parameter("x")
        for j in range(N):
            for r in range(M,2**L):
                circ.compose(spin_one(-2*params["l1"]*dt*x**(params["dynamic_l"]+1),r),list(range(L*j, L*j+L)),inplace=True)
            for b in range(M):
                circ.compose(spin_one(-2*dt*x*(problem.list_query.query(vec_A[j],vec_B[b],problem.same_group_loss,problem.diff_group_loss,)),b)
                            ,list(range(L * j, L * j + L)),inplace=True)
            for i in range(j + 1):
                for b in range(M):
                    if j != i:
                        circ.compose(spin_two(-2*params["l2"]*dt*x**(params["dynamic_l"]+1),b,b),list(range(L*j,L*j+L))+list(range(L*i, L*i+L)),inplace=True)
                    for a in range(M):
                        dist = mat_B[a][b] - mat_A[i][j]
                        if (not dist == 0):
                            if j == i:
                                if a == b: # now a = b, or the factor is zero.
                                    if problem.subgraph:
                                        if dist < 0:
                                            circ.compose(spin_one(-2*dt*x*dist**2,a),list(range(L*j,L*j+L)),inplace=True)
                                    else:
                                        circ.compose(spin_one(-2*dt*x*dist**2,a),list(range(L*j,L*j+L)),inplace=True)
                            else:
                                circ.compose(spin_two(-2*dt*x*dist**2,b,a),list(range(L*j,L*j+L))+list(range(L*i, L*i+L)),inplace=True)
        circ.rx(2*dt*(1-x)*params["b0"], range(N*L))
        return circ

    circ = QuantumCircuit(N * L)

    # Initial state
    circ.x(range(N * L))
    circ.h(range(N * L))

    unit = circuit_unit()

    # with circ.for_loop(range(params["m0"])) as layer:
    # circ.append(unit.assign_parameters({'x': (layer+1)/(params["m0"]+1)}), range(N*L))
    for layer in range(params["m0"]):
        circ.compose(
            unit.assign_parameters({"x": (layer + 1) / (params["m0"] + 1)}),
            inplace=True,
        )

    circ.measure_all()
    return circ


def QA_simulate(problem: basic.Problem, params: dict) -> dict:
    """
    Run the simulation for a noiseless quantum annealing circuit.

    Args:
        mat_A, mat_B, vec_A, vec_B(np.ndarray): Arguments for the optimization problem.
        params(dict): Parameters for circuit and simulation, contaning the following options.
            device (string): Device for AerSimulator, 'CPU' by default.
            silent (bool): Whether print information to console or not, False by default.
            shots (int): Number of running times, 1,000,000 by default.
            m0 (int): Number of layers, 100 by default.
            t0 (float): Total evolution time, 50 by default.
            b0 (float): Transverse magnetic field strength, 1 by default.
            l1,l2 (float): Coefficients for regularization terms, 10 by defalut.
            dynamic_l (int): Use dynamic l(x)=lx^k, 0 for disable, 4 by default.


    Returns:
        `result`(dict): result information.
    """

    default_params = {
        "device": "CPU",
        "silent": False,
        "shots": 1000000,
        "m0": 100,
        "t0": 50,
        "b0": 1,
        "l1": 10,
        "l2": 10,
        "dynamic_l": 4,
    }
    params = {**default_params, **params}

    N = problem.N
    M = problem.M
    L = problem.L

    total_start_time = time.time()

    if not params["silent"]:
        print("------------------")
        print("Starting QA simulation under the following param: ")
        print(params)
        print("We are using", N * L, "qubits, with N =", N, ", M =", M)

    # Build circuit
    if not params["silent"]:
        print("Building circuit...")
    start_time = time.time()
    circ = QA_circuit(problem, params)
    end_time = time.time()
    if not params["silent"]:
        print(f"Finished in {int(1000*(end_time-start_time))} ms")

    # Run simulation
    if not params["silent"]:
        print("Running simulation...")
    start_time = time.time()
    simulator = AerSimulator(device=params['device'], method='statevector')
    # simulator.set_options(precision='single')
    job = simulator.run(circ, shots=params['shots'])
    sim_result = job.result()
    end_time = time.time()
    if not params["silent"]:
        print(f"Finished in {int(1000*(end_time-start_time))} ms")

    # Data processing
    counts = sim_result.get_counts(circ)
    valid_prob = 0
    d_min = 1000
    d_avg = 0
    d_min_cl, num_sol_clas = problem.brutal_force()
    solutions = []
    sol_prob = 0
    for result, count in counts.items():
        f = problem.result_to_f(result)
        valid = problem.valid(f)
        valid_prob += int(valid) * count
        if not valid:
            continue
        d_value = problem.eval_d(f)
        d_avg += d_value * count
        if d_value == d_min_cl:
            solutions += [f]
            sol_prob += count
        if d_value < d_min:
            d_min = d_value
    if valid_prob == 0:
        if(not params['silent']):
            print("No valid solutions. Not even valid!")
        d_avg = -1
        valid_prob = 0
        sol_prob = 0
    else:
        d_avg /= valid_prob
        valid_prob /= params["shots"]
        sol_prob /= params["shots"]
    if not params["silent"]:
        print(
            f"Valid prob: {valid_prob:.3f}, Solution prob: {sol_prob:.6f}, Average d: {d_avg:.3f}, Min d: {d_min:.3f}, Classical Min d: {d_min_cl:.3f}."
        )
    if not params["silent"]:
        print(f"Out of solutions {num_sol_clas}, quantum algorithm found {solutions}.")

    result = {}
    total_end_time = time.time()
    result['time'] = total_end_time - total_start_time
    result['d_avg'] = d_avg
    result['d_min'] = d_min
    result['d_min_cl'] = d_min_cl
    result['valid_prob'] = valid_prob
    result['sol_prob'] = sol_prob
    result['problem'] = problem
    result['solutions'] = solutions
    result['counts'] = counts

    return result


if __name__ == "__main__":
    # Example usage:

    # cha_b = ['C1', 'C2', 'C3', 'C4', 'O1', 'O2']
    # bonds_b = [('C1','C2'), ('C2','C3'), ('C3','C4'),
    #         ('C4','O1'), ('C4','O1'), ('C4','O2')]
    # hydrogen_b = [3, 2, 2, 0, 0, 1]
    # cha_a = ['C1', 'O1', 'O2']
    # bonds_a = [('C1', 'O1'), ('C1', 'O1'), ('C1', 'O2')]
    # hydrogen_a = [0, 0, 1]

    cha_b = ['O1', 'C1', 'O2', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'Cl1', 'Cl2']
    bonds_b = [('O1', 'C1'), ('C1', 'O2'), ('C1', 'O2'), ('C1', 'C2'), ('C2', 'C3'), ('C2', 'C7'), ('C2', 'C7'), ('C3', 'C4'), ('C3', 'C4'), ('C3', 'Cl1'), ('C4', 'C5'), ('C4', 'Cl2'), ('C5', 'C6'), ('C5', 'C6'), ('C6', 'C7')]
    hydrogen_b = [1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
    cha_a = ['O1', 'C1', 'O2', 'C2']
    bonds_a = [('O1', 'C1'), ('C1', 'O2'), ('C1', 'O2'), ('C1', 'C2')]
    hydrogen_a = [1, 0, 0, 0]
    
    mat_B = preprocess.change_to_graph(cha_b, bonds_b, hydrogen_b, 3, 3)
    mat_A = preprocess.change_to_graph(cha_a, bonds_a, hydrogen_a, 3, 3)

    problem = basic.Problem(
        mat_A, mat_B, cha_a, cha_b, same_group_loss=0.2, diff_group_loss=1.0
    )

    QA_simulate(problem, params={'t0':50, 'm0':100})
    
