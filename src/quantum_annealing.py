from qiskit import *
from qiskit.quantum_info import Operator
from qiskit.circuit.library import PhaseGate
from qiskit_aer import *
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error
import numpy as np
from scipy.linalg import svd, polar
from argparse import *
import time

import preprocess
import basic

# get qubit count
parser = ArgumentParser(description="Parser")
parser.add_argument('--t0', type=float)
parser.add_argument('--m0', type=int)
args = parser.parse_args()

# Change this! Remain symmetric please, there can be self-edges.
# # Example 0 from COOH find COOH
# B = -COOH
# cha_b = ['C1', 'O1', 'O2', 'H1']
# bonds_b = [('C1', 'O1'), ('C1', 'O1'), ('C1', 'O2'), ('O2', 'H1')]
cha_b = ['C1', 'C2', 'C3', 'C4', 'O1', 'O2', 'H1', 'H2', 'H3', "H4", 'H5', 'H6', 'H7', 'H8']
bonds_b = [('C1','H1'), ('C2','H2'), ('C3','H3'), ('C1','H4'), 
           ('C1','H5'), ('C2','H6'), ('C3','H7'), 
           ('C1','C2'), ('C2','C3'), ('C3','C4'),
           ('C4','O1'), ('C4','O1'), ('C4','O2'), ('H8','O2')]

# # A = -COOH
cha_a = ['C1', 'O1', 'O2', 'H1']
bonds_a = [('C1', 'O1'), ('C1', 'O1'), ('C1', 'O2'), ('O2', 'H1')]

# A = -CO
# cha_a = ['C1', 'O1', 'O2', 'H1']
# bonds_a = [('C1','O1'), ('C1','O1'), ('C1','O2'), ('H1','O2')]

# embed
embd, B = preprocess.change_to_graph(cha_b, bonds_b)
_, A = preprocess.change_to_graph(cha_a, bonds_a, embd)

N = len(A[0,:])
M = len(B[0,:])

T0 = args.t0 # 40
M0 = args.m0 # 200

L1 = L2 = 10

SHOTS = 1000000

# error. ~99.8% for single qubit gate, ~99% for double qubit gate
# T1 = 200#ns
# T2 = 100#ns
# TAU1 = 0.2
# TAU2 = 1

# Below are the code, do not change.
L = len(bin(M-1)) - 2
print("We are using",N*L,"qubits, with N =",N,", M =",M)

def nearest_unitary(A):
    U, _ = polar(A)
    return U

# def cnphi(n, phi):
#     qc = QuantumCircuit(n)

#     cphase_gate = PhaseGate(phi).control(n-1)
#     qc.append(cphase_gate, range(n))

#     basis_gates = ['u1', 'u2', 'u3', 'id', 'cx']
#     # transpiled_qc = transpile(qc, basis_gates=basis_gates)
#     return qc

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

def apply(Ui,circ,list):
    Ui = nearest_unitary(Ui)
    Ui = Operator(Ui)
    circ.append(Ui,list)

def annealing(T,M0):
    circ = QuantumCircuit(N*L)

    # Initial state
    circ.x(range(N*L))
    circ.h(range(N*L))

    dt = T/M0

    for m in range(M0):
        # print(m)
        for j in range(N):
            for r in range(M,2**L):
                circ.compose(spin_one(-L1*dt*((m+1)/M0)**5,r),list(range(L*j, L*j+L)),inplace=True)
            for i in range(j+1):
                for b in range(M):
                    # if j == i:
                    #     circ.compose(spin_one(1*dt*((m+1)/M0), b), list(range(L*j, L*j+L)),inplace=True) # why?
                    if j != i:
                        circ.compose(spin_two(-L2*dt*((m+1)/M0)**5,b,b),list(range(L*j,L*j+L))+list(range(L*i, L*i+L)),inplace=True)
                    for a in range(M):
                        if (B[a][b] != A[i][j]):
                            if j == i:
                                if a == b: # now a = b, or the factor is zero.
                                    circ.compose(spin_one(-2*dt*((m+1)/M0)*(B[a][b]-A[i][j])**2,a),list(range(L*j,L*j+L)),inplace=True)
                            else:
                                circ.compose(spin_two(-2*dt*((m+1)/M0)*(B[a][b]-A[i][j])**2,b,a),list(range(L*j,L*j+L))+list(range(L*i, L*i+L)),inplace=True)
            Ui = np.eye(2) - 1j*np.array([[1,1],[1,1]])*dt*(1-(m+1)/M0)
            for x in range(N*L):
                apply(Ui,circ,[x])
    circ.measure_all()
    return circ

print("Building circuit...")
start_time = time.time()
circ = annealing(T0, M0)
end_time = time.time()
print(f"Finished in {int(1000*(end_time-start_time))} ms")

print('Running simulation...')
start_time = time.time()
backend = AerSimulator(device='GPU')
job = backend.run(circ, shots=SHOTS)
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
    W_value  = basic.eval_W(f, A, B, L1, L2)
    if W_value == 0:
        solutions += f
        sol_prob += count
    if W_value < W_min:
        W_min = W_value
        f_min = f
    W_avg += W_value * count
valid_prob /= SHOTS
W_avg /= SHOTS
sol_prob /= SHOTS
print(f"Valid prob: {valid_prob:.3f}, Average W: {W_avg:.3f}, Min W: {W_min:.3f}, Solution prob: {sol_prob:.6f}")
# top_5 = sorted_list[:5]
# for array, value in top_5:
#     print(f"Array (flipped): {array}, Freq: {100*value/SHOTS:.3f}%")
    


# # Convert counts to a numpy array of length n
# x_values = []
# for key in counts.keys():
#     x_values += [np.flip(np.array([int(b) for b in format(int(key,base=2), '0'+str(N*L+2)+'b')]))] * counts[key]
# x_values = np.array(x_values)

# n_sol = 0
# last_x = [0]*N*L
# for x_value in x_values:
#     v_values = x_value[:N*L].reshape(N, L)

#     # Calculate the value for each row: 4*v1 + 2*v2 + v3
#     values = [0]*N
#     for i in range(L):
#         values = values + (1<<(L-i-1)) * v_values[:,i]
#     if (np.any(values > M-1)):
#         continue

#     # Create the one-hot encoded matrix
#     one_hot_matrix = np.zeros((N, M))
#     one_hot_matrix[np.arange(N), values] = 1

#     # Calculate PBP^T
#     P = one_hot_matrix
#     PBP_T = P @ B @ np.transpose(P)

#     # Check if PBP^T matches A
#     if np.all(PBP_T == A):
#         if np.any(x_value[:N*L] != last_x):
#             print("solution!", values, x_value)
#             last_x = x_value[:N*L]
#         n_sol = n_sol + 1
# print("# of correct solutions:", n_sol)
