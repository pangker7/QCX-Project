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
def QA_circuit(A, B, params):

    N, M, L = basic.get_NML(A, B)
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
            for i in range(j+1):
                for b in range(M):
                    if j != i:
                        circ.compose(spin_two(-params["l2"]*dt*x**(params["dynamic_l"]+1),b,b),list(range(L*j,L*j+L))+list(range(L*i, L*i+L)),inplace=True)
                    for a in range(M):
                        if (B[a][b] != A[i][j]):
                            if j == i:
                                if a == b: # now a = b, or the factor is zero.
                                    circ.compose(spin_one(-2*dt*x*(B[a][b]-A[i][j])**2,a),list(range(L*j,L*j+L)),inplace=True)
                            else:
                                circ.compose(spin_two(-2*dt*x*(B[a][b]-A[i][j])**2,b,a),list(range(L*j,L*j+L))+list(range(L*i, L*i+L)),inplace=True)
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

def QA_simulate(A, B, params):
    default_params = {'t0': 20, 'm0': 100, 'shots': 1000000, 'l1': 10, 'l2': 10, 'dynamic_l': 3, 'b0': 1}
    params = {**default_params, **params}

    N, M, L = basic.get_NML(A, B)
    print("We are using",N*L,"qubits, with N =",N,", M =",M)

    print("Building circuit...")
    start_time = time.time()
    circ = QA_circuit(A, B, params)
    end_time = time.time()
    print(f"Finished in {int(1000*(end_time-start_time))} ms")

    print('Running simulation...')
    start_time = time.time()
    backend = AerSimulator(device='GPU')
    job = backend.run(circ, shots=params['shots'])
    result = job.result()
    end_time = time.time()
    print(f"Finished in {int(1000*(end_time-start_time))} ms")

    counts = result.get_counts(circ)
    sorted_list = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    valid_prob = 0
    W_min = 1000
    f_min = []
    W_avg = 0
    solutions = []
    sol_prob = 0
    for result, count in sorted_list:
        f = basic.result_to_f(result, A, B)
        valid = basic.valid(f, A, B)
        valid_prob += int(valid) * count
        W_value  = basic.eval_W(f, A, B, params['l1'], params['l2'])
        if W_value == 0:
            solutions += f
            sol_prob += count
        if W_value < W_min:
            W_min = W_value
            f_min = f
        W_avg += W_value * count
    valid_prob /= params['shots']
    W_avg /= params['shots']
    sol_prob /= params['shots']
    print(f"Valid prob: {valid_prob:.3f}, Average W: {W_avg:.3f}, Min W: {W_min:.3f}, Solution prob: {sol_prob:.6f}")



if __name__ == "__main__":
    # Example usage:

    cha_b = ['C1', 'C2', 'C3', 'C4', 'O1', 'O2', 'H1', 'H2', 'H3', "H4", 'H5', 'H6', 'H7', 'H8']
    bonds_b = [('C1','H1'), ('C2','H2'), ('C3','H3'), ('C1','H4'), 
            ('C1','H5'), ('C2','H6'), ('C3','H7'), 
            ('C1','C2'), ('C2','C3'), ('C3','C4'),
            ('C4','O1'), ('C4','O1'), ('C4','O2'), ('H8','O2')]
    cha_a = ['C1', 'O1', 'O2', 'H1']
    bonds_a = [('C1', 'O1'), ('C1', 'O1'), ('C1', 'O2'), ('O2', 'H1')]
    
    embd, B = preprocess.change_to_graph(cha_b, bonds_b)
    _, A = preprocess.change_to_graph(cha_a, bonds_a, embd)

    QA_simulate(A, B, params={'t0': 80, 'm0': 400})
    