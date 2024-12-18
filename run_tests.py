import json
from src.qa import QA_simulate
from src.basic import Problem

if __name__ == "__main__":
    with open("./tests/test_cases.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    data = data[-2:]

    for i, p in enumerate(data):
        problem = Problem.from_bonds(p['cha_a'], p['cha_b'], p["bonds_a"], p["bonds_b"], p["hydrogen_a"], p["hydrogen_b"])
        QA_simulate(problem, params={'t0':50, 'm0':100, 'device':'GPU'})
