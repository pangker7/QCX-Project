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

    x = np.fromstring("0.56391455 0.32282257 0.32387736 0.30048132 0.26995095 0.22913798 0.23705612 0.18940412 0.02746699 0.12029765 0.09786844 0.23912383 0.33232745 0.34078035 0.41883407 0.46104773 0.47693149 0.55881904 0.33126118 0.46475805 0.17852747 0.3481353  0.41897482 0.59146415 0.66328601 0.68029877 0.75392305 1.03072926 1.59612783 2.51783015", sep=' ')
    np.save("model/10_1218a.npy", x)
    x0 = np.zeros(60)
    for i in range(30):
        x0[2*i] = x[i]
        x0[2*i+1] = x[i]
    x1 = QVA_optimize(problem, x0=x0, params={'lr':0.1, 'epochs':40, 'm0':20}, loss_func='sol_prob')
    np.save("model/20_1219a.npy", x1)