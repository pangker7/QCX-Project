from src.basic import Problem
from src.qva import QVA_optimize
import numpy as np

if __name__ == "__main__":

    cha_b = ['O1', 'O2', 'O3', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
    bonds_b = [('O1', 'O2'), ('O1', 'C1'), ('O2', 'C6'), ('O3', 'C6'), ('O3', 'C6'), ('C1', 'C2'), ('C1', 'C3'), ('C1', 'C4'), ('C5', 'C6'), ('C5', 'C7'), ('C5', 'C8')]
    hydrogen_b = [0, 0, 0, 0, 3, 3, 3, 1, 0, 3, 3]
    cha_a = ['O1', 'C1', 'O2', 'C2']
    bonds_a = [('O1', 'C1'), ('C1', 'O2'), ('C1', 'O2'), ('C1', 'C2')]
    hydrogen_a = [1, 0, 0, 0]

    problem = Problem.from_bonds(cha_a, cha_b, bonds_a, bonds_b, hydrogen_a, hydrogen_b)

    x1 = QVA_optimize(problem, x0=None, params={'lr':0.1, 'epochs':10, 'm0':20}, loss_func='w_avg')
    np.savetxt("model/20_eg1.txt", x1)
    x2 = QVA_optimize(problem, x0=x1, params={'lr':0.1, 'epochs':10, 'm0':20}, loss_func='sol_prob')
    np.savetxt("model/20_eg2.txt", x2)