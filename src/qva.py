from qiskit import *
from qiskit.circuit import Parameter, ParameterVector
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

def QVA_circuit(problem:basic.Problem, params:dict):
    """
    Construct the quantum circuit for QVA using QAOA ansatz
    """

    mat_A = problem.mat_A
    mat_B = problem.mat_B
    vec_A = problem.vec_A
    vec_B = problem.vec_B
    N = problem.N
    M = problem.M
    L = problem.L
    m0 = params['m0']

    def spin_one(phi,n):
        this = QuantumCircuit(L)
        bin_n = bin(n)[2:] 
        for m in range(L):
            if len(bin_n) <= m or bin_n[-m-1] == '0':
                this.x(L-1-m)
        if (L > 1):
            this.compose(PhaseGate(phi).control(L-1),range(L),inplace=True)
        else:
            this.compose(PhaseGate(phi),range(L),inplace=True)
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
        t_bx = Parameter('t_bx')        # B_x
        t_func = Parameter('t_func')    # target function
        t_reg = Parameter('t_reg')      # regularization
        for j in range(N):
            for r in range(M,2**L):
                circ.compose(spin_one(-2*t_reg,r),list(range(L*j, L*j+L)),inplace=True)
            for b in range(M):
                dist = problem.list_query.query(vec_A[j],vec_B[b],problem.same_group_loss,problem.diff_group_loss)
                if dist > 0:
                    circ.compose(spin_one(-2*t_func*dist,b),list(range(L*j,L*j+L)),inplace=True)
            for i in range(j+1):
                for b in range(M):
                    if j != i:
                        circ.compose(spin_two(-2*t_reg,b,b),list(range(L*j,L*j+L))+list(range(L*i, L*i+L)),inplace=True)
                    for a in range(M):
                        dist = mat_B[a][b] - mat_A[i][j]
                        if (not dist == 0):
                            if j == i:
                                if a == b: # now a = b, or the factor is zero.
                                    if problem.subgraph:
                                        if dist < 0:
                                            circ.compose(spin_one(-2*t_func*dist**2,a),list(range(L*j,L*j+L)),inplace=True)
                                    else:
                                        circ.compose(spin_one(-2*t_func*dist**2,a),list(range(L*j,L*j+L)),inplace=True)
                            else:
                                circ.compose(spin_two(-2*t_func*dist**2,b,a),list(range(L*j,L*j+L))+list(range(L*i, L*i+L)),inplace=True)
        circ.rx(2*t_bx, range(N*L))
        return circ
    
    circ = QuantumCircuit(N*L)

    m0 = params['m0']
    vt_bx = ParameterVector('vt_bx', m0)        # B_x
    vt_func = ParameterVector('vt_func', m0)    # target function
    vt_reg = ParameterVector('vt_reg', m0)      # regularization

    # Initial state
    circ.x(range(N*L))
    circ.h(range(N*L))

    unit = circuit_unit()

    for layer in range(m0):
        circ.compose(unit.assign_parameters({'t_bx': vt_bx.params[layer], 't_func': vt_func.params[layer], 't_reg': vt_reg.params[layer]}),inplace=True)

    circ.measure_all()
    return circ


def QVA_run(circ:QuantumCircuit, x:np.ndarray, params:dict):
    """
    Run the simulation for the QVA circuit to evaluate target function d
    """

    circ = circ.assign_parameters(x, inplace=False)
    simulator = AerSimulator(device=params['device'], method='statevector')
    job = simulator.run(circ, shots=params['shots'])
    sim_result = job.result()

    counts = sim_result.get_counts(circ)
    valid_prob = 0
    d_min = 1000
    d_avg = 0
    d_min_cl, cl_sols = problem.cl_solution
    w_min = 1000
    w_avg = 0
    w_hist = {}
    solutions = []
    sol_prob = 0
    for result, count in counts.items():
        f = problem.result_to_f(result)
        valid = problem.valid(f)
        valid_prob += int(valid) * count
        w_value = problem.eval_W(f, params['l1'], params['l2'])
        w_avg += w_value * count
        w_hist.setdefault(w_value, 0)
        w_hist[w_value] += count
        if w_value < w_min:
            w_min = w_value
        if not valid:
            continue
        d_value  = problem.eval_d(f)
        d_avg += d_value * count
        if d_value == d_min_cl:
            solutions += [f]
            sol_prob += count
        if d_value < d_min:
            d_min = d_value
    w_avg /= params['shots']
    w_hist = sorted(w_hist.items(), key=lambda p : p[0])
    if valid_prob == 0:
        d_avg = -1
        valid_prob = 0
        sol_prob = 0
    else:
        d_avg /= valid_prob
        valid_prob /= params['shots']
        sol_prob /= params['shots']
    
    result = {}
    result['w_avg'] = w_avg
    result['w_min'] = w_min
    result['w_hist'] = w_hist
    result['d_avg'] = d_avg
    result['d_min'] = d_min
    result['valid_prob'] = valid_prob
    result['sol_prob'] = sol_prob
    result['solutions'] = solutions

    return result


def QVA_optimize(problems:list[basic.Problem], params:dict={}) -> dict:
    """
    Run optimization for QVA circuits for a set of problems and return the optimized circuit parameters.
    Args:
        problems (list[basic.Problem]): List of problems to optimize.
        params (dict): Parameters for circuit and simulation, contaning the following options.
            device (string): Device for AerSimulator, 'CPU' by default.
            silent (bool): Whether print information to console or not, False by default.
            shots (int): Number of running times for each circuit, 1,000,000 by default.
            epochs (int): Number of optimization ephochs
            lr (float): learning rate for gradient decent
            m0 (int): Number of layers, 5 by default.
            l1,l2 (float): Coefficients for regularization terms, 10 by defalut.

    Returns:
        `x (np.ndarray): optimized circuit parameters.
    """

    default_params = {'device': 'CPU', 'silent': False, 'shots': 1000000, 
                      'epochs': 10, 'lr': 0.1, 'm0': 10, 'l1': 2, 'l2': 2}
    params = {**default_params, **params}

    N = problem.N
    M = problem.M
    L = problem.L

    if(not params['silent']):
        print("------------------")
        print("Starting QAOA optimization under the following param: ")
        print(params)
        print("We are using",N*L,"qubits, with N =",N,", M =",M)

    # Build circuit
    if(not params['silent']):
        print("Building circuit...")
    start_time = time.time()
    circ = QVA_circuit(problem, params)
    end_time = time.time()
    if(not params['silent']):
        print(f"Finished in {int(1000*(end_time-start_time))} ms")

    # Run optimization
    if(not params['silent']):
        print('Running optimization...')

    # Inital params
    m0 = params['m0']
    dt = 0.5
    vt_bx = dt * np.array(range(m0)[::-1]) / m0
    vt_func = dt * np.array(range(1, m0+1)) / m0
    vt_reg = 5 * dt * (np.array(range(1, m0+1)) / m0) ** 4
    x0 = np.concatenate((vt_bx, vt_func, vt_reg))
    x0 = np.fromstring("0.50088642  0.26506707  0.25897509  0.23941262  0.19595158  0.16509826  0.19426582  0.18108924  0.05771234  0.06291906 -0.02278328  0.0786359  0.16914648  0.19343238  0.24072872  0.24033925  0.31208672  0.46683556  0.32480595  0.43947762  0.03918165  0.20645738  0.34412306  0.42481009  0.48928957  0.49324575  0.69216613  1.00998438  1.51448759  2.47851422", sep='  ')
    if(not params['silent']):
        print('Initial parameters: ', x0)
    result0 = QVA_run(circ, x0, params)
    print(result0)
    return 0

    # GD
    num_params = len(x0)
    delta = 0.01
    lr = params['lr']
    grad = np.zeros(num_params)
    w_avg = []
    valid_prob = []
    sol_prob = []
    w_avg.append(result0['w_avg'])
    valid_prob.append(result0['valid_prob'])
    sol_prob.append(result0['sol_prob'])
    for epoch in range(params['epochs']):
        print(f"--- Epoch {epoch} ---")
        for i in range(num_params):
            x1 = x0.copy()
            x1[i] += delta
            result1 = QVA_run(circ, x1, params)
            grad[i] = (result1['w_avg']-result0['w_avg']) / delta 
        print(grad)

        t_l = 0
        t_r = 1
        w_l = result0['w_avg']
        w_r = QVA_run(circ, x0-t_r*lr*grad, params)['w_avg']
        # print(f"{t_l}: {w_l}")
        # print(f"{t_r}: {w_r}")
        for _ in range(6):
            t_m = (t_r + t_l) / 2
            w_m = QVA_run(circ, x0-t_m*lr*grad, params)['w_avg']
            # print(f"{t_m}: {w_m}")
            if w_m > w_l:
                t_r = t_m
                w_r = w_m
            elif w_m < w_l and w_m > w_r:
                t_l = t_m
                w_l = w_m
            else:
                t_lm = (t_l + t_m) / 2
                w_lm = QVA_run(circ, x0-t_lm*lr*grad, params)['w_avg']
                if w_lm < w_l and w_lm < w_m:
                    t_r = t_m
                    w_r = w_m
                else:
                    t_l = t_m
                    w_l = w_m

        x0 = x0 - t_m * lr * grad
        result0 = QVA_run(circ, x0, params)
        w_avg.append(result0['w_avg'])
        valid_prob.append(result0['valid_prob'])
        sol_prob.append(result0['sol_prob'])
        print(t_m)
        print(x0)
        print(result0)
    print(w_avg)
    print(valid_prob)
    print(sol_prob)

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

    problem = basic.Problem(mat_A, mat_B, cha_a, cha_b, same_group_loss=0.2, diff_group_loss=1.0)

    QVA_optimize(problem)
    