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

import preprocess
import basic

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

    dt = params["t0"]/params["m0"]

    def spin_one(phi,n):
        this = QuantumCircuit(L)
        bin_n = bin(n)[2:]
        for m in range(L):
            if len(bin_n) <= m or bin_n[-m-1] == '0':
                this.x(L-1-m)
        this.compose(PhaseGate(phi).control(L-1),range(L),inplace=True)
        for m in range(L):
            if len(bin_n) <= m or bin_n[-m-1] == '0':
                this.x(L-1-m)
        return this

    def spin_two(phi,n,m):
        this = QuantumCircuit(2*L)
        bin_n = bin(n)[2:]
        for mu in range(L):
            if len(bin_n) <= mu or bin_n[-mu-1] == '0':
                this.x(L-1-mu)
        bin_m = bin(m)[2:]
        for mu in range(L):
            if len(bin_m) <= mu or bin_m[-mu-1] == '0':
                this.x(2*L-1-mu)
        this.compose(PhaseGate(phi).control(2*L-1),range(2*L),inplace=True)
        for mu in range(L):
            if len(bin_n) <= mu or bin_n[-mu-1] == '0':
                this.x(L-1-mu)
        for mu in range(L):
            if len(bin_m) <= mu or bin_m[-mu-1] == '0':
                this.x(2*L-1-mu)
        return this

    def circuit_unit():
        circ = QuantumCircuit(N*L)
        x = Parameter('x')
        for j in range(N):
            for r in range(M,2**L):
                circ.compose(spin_one(-params["l1"]*dt*x**(params["dynamic_l"]+1),r),list(range(L*j, L*j+L)),inplace=True)
            for b in range(M):
                circ.compose(spin_one(-2*dt*x*(problem.list_query.query(vec_A[j],vec_B[b],problem.same_group_loss,problem.diff_group_loss)),b),list(range(L*j,L*j+L)),inplace=True)
            for i in range(j+1):
                for b in range(M):
                    if j != i:
                        circ.compose(spin_two(-params["l2"]*dt*x**(params["dynamic_l"]+1),b,b),list(range(L*j,L*j+L))+list(range(L*i, L*i+L)),inplace=True)
                    for a in range(M):
                        if (mat_B[a][b] != mat_A[i][j]):
                            if j == i:
                                if a == b: # now a = b, or the factor is zero.
                                    circ.compose(spin_one(-2*dt*x*(mat_B[a][b]-mat_A[i][j])**2,a),list(range(L*j,L*j+L)),inplace=True)
                            else:
                                circ.compose(spin_two(-2*dt*x*(mat_B[a][b]-mat_A[i][j])**2,b,a),list(range(L*j,L*j+L))+list(range(L*i, L*i+L)),inplace=True)
        circ.rx(2*dt*(1-x)*params["b0"], range(N*L))
        return circ
    
    circ = QuantumCircuit(N*L)

    # Initial state
    circ.x(range(N*L))
    circ.h(range(N*L))

    unit = circuit_unit()

    # with circ.for_loop(range(params["m0"])) as layer:
        # circ.append(unit.assign_parameters({'x': (layer+1)/(params["m0"]+1)}), range(N*L))
    for layer in range(params["m0"]):
        circ.compose(unit.assign_parameters({'x': (layer+1)/(params["m0"]+1)}),inplace=True)

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

    default_params = {'device': 'CPU', 'silent': False, 'shots': 1000000, 
                      'm0': 100, 't0': 50, 'b0': 1, 'l1': 10, 'l2': 10, 'dynamic_l': 4}
    params = {**default_params, **params}

    mat_A = problem.mat_A
    mat_B = problem.mat_B
    vec_A = problem.vec_A
    vec_B = problem.vec_B
    N = problem.N
    M = problem.M
    L = problem.L

    total_start_time = time.time()

    if(not params['silent']):
        print("------------------")
        print("Starting QA simulation under the following param: ")
        print(params)
        print("We are using",N*L,"qubits, with N =",N,", M =",M)

    # Build circuit
    if(not params['silent']):
        print("Building circuit...")
    start_time = time.time()
    circ = QA_circuit(problem, params)
    end_time = time.time()
    if(not params['silent']):
        print(f"Finished in {int(1000*(end_time-start_time))} ms")

    # Run simulation
    if(not params['silent']):
        print('Running simulation...')
    start_time = time.time()
    backend = AerSimulator(device=params['device'])
    job = backend.run(circ, shots=params['shots'])
    sim_result = job.result()
    end_time = time.time()
    if(not params['silent']):
        print(f"Finished in {int(1000*(end_time-start_time))} ms")

    # Data processing
    counts = sim_result.get_counts(circ)
    sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    valid_prob = 0
    d_min = 1000
    d_avg = 0
    d_min_cl, _ = problem.brutal_force()
    solutions = []
    sol_prob = 0
    for result, count in sorted_counts:
        f = problem.result_to_f(result)
        valid = problem.valid(f)
        valid_prob += int(valid) * count
        if not valid:
            continue
        d_value  = problem.eval_d(f)
        d_avg += d_value * count
        if d_value == d_min_cl:
            solutions += f
            sol_prob += count
        if d_value < d_min:
            d_min = d_value
    d_avg /= valid_prob
    valid_prob /= params['shots']
    sol_prob /= params['shots']
    if(not params['silent']):
        print(f"Valid prob: {valid_prob:.3f}, Solution prob: {sol_prob:.6f}, Average d: {d_avg:.3f}, Min d: {d_min:.3f}, Classical Min d: {d_min_cl:.3f}.")
    
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
    result['counts'] = sorted_counts

    return result


if __name__ == "__main__":
    # Example usage:

    cha_b = ['C1', 'C2', 'C3', 'C4', 'O1', 'O2']
    bonds_b = [('C1','C2'), ('C2','C3'), ('C3','C4'),
            ('C4','O1'), ('C4','O1'), ('C4','O2')]
    hydrogen_b = [3, 2, 2, 0, 0, 1]
    cha_a = ['C1', 'O1', 'O2']
    bonds_a = [('C1', 'O1'), ('C1', 'O1'), ('C1', 'O2')]
    hydrogen_a = [0, 0, 1]
    
    mat_B = preprocess.change_to_graph(cha_b, bonds_b, hydrogen_b, 3, 3)
    mat_A = preprocess.change_to_graph(cha_a, bonds_a, hydrogen_a, 3, 3)

    problem = basic.Problem(mat_A, mat_B, cha_a, cha_b, same_group_loss=0.2, diff_group_loss=1.0)

    QA_simulate(problem, params={'t0': 50, 'm0': 100})
    