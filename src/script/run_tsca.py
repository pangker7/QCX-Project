import json
from re import sub
from ..qa import QA_simulate
from ..basic import Problem
import pandas as pd
import csv
from tqdm import tqdm

# row: chemSpiderId, len_molecule, prob_valid, prob_solution, d_min, d_min_cl, find_rate=len(solutions)/len(solutions_cl)
# column: molecule_id

# (group_idx, group_size): (0,1), (6,1), (2,2), (24,2), (11,3), (15,3), (25,6), (16,5)


def convert_str_to_list_or_tuple(val):
    try:
        return eval(val)
    except:
        return val


df_group = pd.read_csv("./data/fun_group.csv")
group_strictly_equal = df_group["StrictlyEqual"]
group_vertice_set = df_group["VerticeSet"].apply(convert_str_to_list_or_tuple)
group_edge_set = df_group["EdgeSet"].apply(convert_str_to_list_or_tuple)
group_num_hydrogen_set = df_group["NumHydrogenSet"].apply(convert_str_to_list_or_tuple)

df_molecule = pd.read_csv("./data/molecule_info.csv")
molecule_vertice_set = df_molecule["VerticeSet"].apply(convert_str_to_list_or_tuple)
molecule_spider_id = df_molecule["ChemSpiderID"].apply(convert_str_to_list_or_tuple)
molecule_edge_set = df_molecule["EdgeSet"].apply(convert_str_to_list_or_tuple)
molecule_num_hydrogen_set = df_molecule["NumHydrogenSet"].apply(convert_str_to_list_or_tuple)

with open("./data/TSCA_run_data.json", "r", encoding="utf-8") as file:
    run_data = json.load(file)

group_idx = 16
run_data_group = run_data[group_idx]
group_id = run_data_group["group_id"]
group_name = run_data_group["group_name"]
molecule_id_list = run_data_group["molecule_id_list"]


cha_a = group_vertice_set[group_idx]
bonds_a = group_edge_set[group_idx]
hydrogen_a = group_num_hydrogen_set[group_idx]
subgraph = not bool(group_strictly_equal[group_idx])


# write field_name
field_name = [
    "molecule_id",
    "chemSpiderId",
    "len_molecule",
    "valid_prob",
    "sol_prob",
    "d_min",
    "d_min_cl",
    "degeneracy",
    "find_rate",
]
with open(f"./data/TSCA_result/{group_idx+1}_{group_name}.csv", mode="a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(field_name)

molecule_id_list = molecule_id_list[499:]
for molecule_id in tqdm(molecule_id_list):
    cha_b = molecule_vertice_set[molecule_id]
    bonds_b = molecule_edge_set[molecule_id]
    hydrogen_b = molecule_num_hydrogen_set[molecule_id]
    spider_id = molecule_spider_id[molecule_id]

    problem = Problem.from_bonds(cha_a, cha_b, bonds_a, bonds_b, hydrogen_a, hydrogen_b, subgraph=subgraph)
    result = QA_simulate(problem, params={"t0": 50, "m0": 100, "device": "GPU"})

    d_min_cl, solutions_cl = problem.cl_solution
    degeneracy = len(solutions_cl)
    find_rate = len(result["solutions"]) / degeneracy

    row = [
        molecule_id,
        spider_id,
        len(cha_b),
        result["valid_prob"],
        result["sol_prob"],
        result["d_min"],
        result["d_min_cl"],
        degeneracy,
        find_rate,
    ]
    with open(f"./data/TSCA_result/{group_idx+1}_{group_name}.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(row)
