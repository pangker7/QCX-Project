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
    Run optimization for QVA circuit for a set of problems and return the optimized circuit parameters.
    Args:
        problems (list[basic.Problem]): List of problems to optimize.
        params (dict): Parameters for circuit and simulation, contaning the following options.
            device (string): Device for AerSimulator, 'CPU' by default.
            silent (bool): Whether print information to console or not, False by default.
            shots (int): Number of running times for each circuit, 1,000,000 by default.
            epochs (int): Number of optimization ephochs, 10 by default.
            lr (float): Learning rate for gradient decent, 0.1 by default.
            m0 (int): Number of layers, 10 by default.
            l1,l2 (float): Coefficients for regularization terms, 2 by defalut.

    Returns:
        `x (np.ndarray): Optimized circuit parameters.
    """

    default_params = {'device': 'CPU', 
                      'silent': False, 
                      'shots': 1000000, 
                      'epochs': 10, 
                      'lr': 0.1, 
                      'm0': 10, 
                      'l1': 2, 
                      'l2': 2}
    params = {**default_params, **params}

    N = problem.N
    M = problem.M
    L = problem.L

    if(not params['silent']):
        print("------------------")
        print("Starting QVA optimization under the following param: ")
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
    start_time = time.time()

    # loss function
    def loss(x, output=False):
        result = QVA_run(circ, x, params)
        # The loss function can be set to w_avg or valid_prob+sol_prob
        # loss = result['w_avg']
        loss = -result['valid_prob']-50*result['sol_prob']
        if output:
            print(f"Loss:{loss:.6f}, Valid prob: {result['valid_prob']:.6f}, Solution prob: {result['sol_prob']:.6f}, Average W: {result['w_avg']:.6f}.")
        return loss, result
    
    # Inital params
    m0 = params['m0']
    dt = 0.5
    vt_bx = dt * np.array(range(m0)[::-1]) / m0
    vt_func = dt * np.array(range(1, m0+1)) / m0
    vt_reg = 5 * dt * (np.array(range(1, m0+1)) / m0) ** 4
    x0 = np.concatenate((vt_bx, vt_func, vt_reg))
    # x0 = np.fromstring("0.50088642  0.26506707  0.25897509  0.23941262  0.19595158  0.16509826  0.19426582  0.18108924  0.05771234  0.06291906 -0.02278328  0.0786359  0.16914648  0.19343238  0.24072872  0.24033925  0.31208672  0.46683556  0.32480595  0.43947762  0.03918165  0.20645738  0.34412306  0.42481009  0.48928957  0.49324575  0.69216613  1.00998438  1.51448759  2.47851422", sep='  ')
    # x0 = np.fromstring("0.55567744 0.31985418 0.3154204 0.29289699 0.24580314 0.25377599 0.26581269 0.25639002 0.01231195 0.12632531 0.02566008 0.25065348 0.30052734 0.29431129 0.31064278 0.35470253 0.42783867 0.56750939 0.32998368 0.48692684 0.19225587 0.23341441 0.38693361 0.54335892 0.62600832 0.60938638 0.71355285 1.08386133 1.52981962 2.54265875", sep=' ')
    # x0 = np.fromstring("0.58141084 0.35431984 0.32296056 0.3045198  0.26732146 0.29228505 0.26268464 0.21704494 0.00885976 0.12957597 0.04933953 0.27064219 0.35276554 0.39305281 0.39410575 0.4071655  0.44757094 0.58181396 0.30172231 0.44724496 0.18634681 0.31204253 0.38848888 0.57642626 0.68998328 0.66488236 0.73852051 1.03074391 1.5579331  2.50588777", sep=' ')
    # x0 = np.fromstring("0.56391455 0.32282257 0.32387736 0.30048132 0.26995095 0.22913798 0.23705612 0.18940412 0.02746699 0.12029765 0.09786844 0.23912383 0.33232745 0.34078035 0.41883407 0.46104773 0.47693149 0.55881904 0.33126118 0.46475805 0.17852747 0.3481353  0.41897482 0.59146415 0.66328601 0.68029877 0.75392305 1.03072926 1.59612783 2.51783015", sep=' ')
    if(not params['silent']):
        print('Initial parameter x = ', x0)
    loss0, result0 = loss(x0, output=not params["silent"])
    losses = [loss0]
    w_avgs = [result0['w_avg']]
    valid_probs = [result0['valid_prob']]
    sol_probs = [result0['sol_prob']]

    # GD
    num_params = len(x0)
    delta = 0.01
    lr = params['lr']
    grad = np.zeros(num_params)

    for epoch in range(params['epochs']):
        if(not params['silent']):
            print(f"--- Epoch {epoch} ---")
        for i in range(num_params):
            x1 = x0.copy()
            x1[i] += delta
            loss1, _ = loss(x1)
            grad[i] = (loss1 - loss0) / delta 
        print(grad.size)
        if(not params['silent']):
            print(f"grad = {grad}")

        # Binary search for best lr
        t_l = 0
        t_r = 1
        loss_l = loss0
        loss_r, _ = loss(x0-t_r*lr*grad)
        for _ in range(8):
            t_m = (t_r + t_l) / 2
            loss_m, _ = loss(x0-t_m*lr*grad)
            if loss_m > loss_l:
                t_r = t_m
                loss_r = loss_m
            elif loss_m < loss_l and loss_m > loss_r:
                t_l = t_m
                loss_l = loss_m
            else:
                t_lm = (t_l + t_m) / 2
                loss_lm, _ = loss(x0-t_lm*lr*grad)
                if loss_lm < loss_l and loss_lm < loss_m:
                    t_r = t_m
                    loss_r = loss_m
                else:
                    t_l = t_m
                    loss_l = loss_m

        x0 = x0 - t_m*lr*grad
        if(not params['silent']):
            print(f"t = {t_m}")
            print(f"Parameter x = {x0}")
        loss0, result0 = loss(x0, output=not params["silent"])
        losses.append(loss0)
        w_avgs.append(result0['w_avg'])
        valid_probs.append(result0['valid_prob'])
        sol_probs.append(result0['sol_prob'])
        end_time = time.time()
    if(not params['silent']):
        print(f"Finished in {int(1000*(end_time-start_time))} ms")
        print(f"Final parameter x = {x0}")
        print(f"loss = {losses}")
        print(f"w_avgs = {w_avgs}")
        print(f"valid_probs = {valid_probs}")
        print(f"sol_probs = {sol_probs}")

if __name__ == "__main__":
    # Example usage for training:

    cha_b = ['C1', 'C2', 'C3', 'C4', 'O1', 'O2']
    bonds_b = [('C1','C2'), ('C2','C3'), ('C3','C4'),
            ('C4','O1'), ('C4','O1'), ('C4','O2')]
    hydrogen_b = [3, 2, 2, 0, 0, 1]
    cha_a = ['C1', 'O1', 'O2']
    bonds_a = [('C1', 'O1'), ('C1', 'O1'), ('C1', 'O2')]
    hydrogen_a = [0, 0, 1]

    # cha_b = ['O1', 'C1', 'O2', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'Cl1', 'Cl2']
    # bonds_b = [('O1', 'C1'), ('C1', 'O2'), ('C1', 'O2'), ('C1', 'C2'), ('C2', 'C3'), ('C2', 'C7'), ('C2', 'C7'), ('C3', 'C4'), ('C3', 'C4'), ('C3', 'Cl1'), ('C4', 'C5'), ('C4', 'Cl2'), ('C5', 'C6'), ('C5', 'C6'), ('C6', 'C7')]
    # hydrogen_b = [1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
    # cha_a = ['O1', 'C1', 'O2', 'C2']
    # bonds_a = [('O1', 'C1'), ('C1', 'O2'), ('C1', 'O2'), ('C1', 'C2')]
    # hydrogen_a = [1, 0, 0, 0]

    # cha_b = ['O1', 'O2', 'O3', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
    # bonds_b = [('O1', 'O2'), ('O1', 'C1'), ('O2', 'C6'), ('O3', 'C6'), ('O3', 'C6'), ('C1', 'C2'), ('C1', 'C3'), ('C1', 'C4'), ('C5', 'C6'), ('C5', 'C7'), ('C5', 'C8')]
    # hydrogen_b = [0, 0, 0, 0, 3, 3, 3, 1, 0, 3, 3]
    # cha_a = ['O1', 'C1', 'O2', 'C2']
    # bonds_a = [('O1', 'C1'), ('C1', 'O2'), ('C1', 'O2'), ('C1', 'C2')]
    # hydrogen_a = [1, 0, 0, 0]

    
    mat_B = preprocess.change_to_graph(cha_b, bonds_b, hydrogen_b, 3, 3)
    mat_A = preprocess.change_to_graph(cha_a, bonds_a, hydrogen_a, 3, 3)

    problem = basic.Problem(mat_A, mat_B, cha_a, cha_b)

    QVA_optimize(problem, params={'lr':0.5, 'epochs':20})
    