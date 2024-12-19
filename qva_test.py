import json
from src.qva import QVA_apply
from src.basic import Problem
import numpy as np

if __name__ == "__main__":
    
    x = np.fromstring("0.56391455 0.32282257 0.32387736 0.30048132 0.26995095 0.22913798 0.23705612 0.18940412 0.02746699 0.12029765 0.09786844 0.23912383 0.33232745 0.34078035 0.41883407 0.46104773 0.47693149 0.55881904 0.33126118 0.46475805 0.17852747 0.3481353  0.41897482 0.59146415 0.66328601 0.68029877 0.75392305 1.03072926 1.59612783 2.51783015", sep=' ')

    with open("./tests/test_cases.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    for i, p in enumerate(data):
        problem = Problem.from_bonds(p['cha_a'], p['cha_b'], p["bonds_a"], p["bonds_b"], p["hydrogen_a"], p["hydrogen_b"])
        QVA_apply(problem, x, params={})
