import json
from re import sub
from src.qva import QVA_apply
from src.basic import Problem
import pandas as pd
import csv
from tqdm import tqdm
import numpy as np

# row: chemSpiderId, len_molecule, prob_valid, prob_solution, d_min, d_min_cl, find_rate=len(solutions)/len(solutions_cl)
# column: molecule_id

# (group_idx, group_size): (0,1), (6,1), (2,2), (24,2), (11,3), (15,3), (25,6), (16,5)

if __name__ == "__main__":

    x = np.fromstring("0.62743267 0.35304619 0.30703319 0.32639527 0.28574767 0.30201025 0.28942999 0.27344163 0.25542818 0.22982751 0.22842361 0.22178032 0.22618143 0.19650389 0.15185053 0.0771635  0.06616679 0.07003316 0.04102851 0.08076335 0.02417664 0.09335289 0.2568116  0.32115883 0.44756491 0.46914862 0.55362719 0.53439894 0.57921868 0.57731098 0.58631707 0.60768597 0.61894758 0.66940266 0.63655791 0.69819431 0.5112777  0.44502798 0.56954301 0.61881356 0.11467145 0.22785188 0.37832214 0.4955053  0.55363177 0.58456552 0.6418963  0.63448227 0.64467249 0.69779601 0.74107002 0.77686498 0.85127594 0.83941875 1.07203051 1.0909286  1.67315013 1.75634416 2.59539648 2.58966144", sep=' ')
    print(x)

    def convert_str_to_list_or_tuple(val):
        try:
            return eval(val)
        except:
            return val

    df_group = pd.read_csv("./data/fun_group.csv")
    group_strictly_equal = df_group["StrictlyEqual"]
    group_vertice_set = df_group["VerticeSet"].apply(convert_str_to_list_or_tuple)
    group_edge_set = df_group["EdgeSet"].apply(convert_str_to_list_or_tuple)
    group_num_hydrogen_set = df_group["NumHydrogenSet"].apply(
        convert_str_to_list_or_tuple
    )

    df_molecule = pd.read_csv("./data/molecule_info.csv")
    molecule_vertice_set = df_molecule["VerticeSet"].apply(convert_str_to_list_or_tuple)
    molecule_spider_id = df_molecule["ChemSpiderID"].apply(convert_str_to_list_or_tuple)
    molecule_edge_set = df_molecule["EdgeSet"].apply(convert_str_to_list_or_tuple)
    molecule_num_hydrogen_set = df_molecule["NumHydrogenSet"].apply(
        convert_str_to_list_or_tuple
    )

    with open("./data/TSCA_run_data.json", "r", encoding="utf-8") as file:
        run_data = json.load(file)

    # (group_idx, group_size): (0,1), (6,1), (2,2), (24,2), (11,3), (15,3), (25,6), (16,5)
    group_idx = 2
    run_data_group = run_data[group_idx]
    group_id = run_data_group["group_id"]
    group_name = run_data_group["group_name"]
    molecule_id_list = run_data_group["molecule_id_list"]

    cha_a = group_vertice_set[group_idx]
    bonds_a = group_edge_set[group_idx]
    hydrogen_a = group_num_hydrogen_set[group_idx]
    subgraph = not group_strictly_equal[group_idx]

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
    with open(f"./data/qva_TSCA_result/{group_idx+1}_{group_name}.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(field_name)

    for molecule_id in tqdm(molecule_id_list):
        cha_b = molecule_vertice_set[molecule_id]
        bonds_b = molecule_edge_set[molecule_id]
        hydrogen_b = molecule_num_hydrogen_set[molecule_id]
        spider_id = molecule_spider_id[molecule_id]

        problem = Problem.from_bonds(
            cha_a, cha_b, bonds_a, bonds_b, hydrogen_a, hydrogen_b, subgraph=subgraph
        )
        result = QVA_apply(problem, x, params={})

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
        with open(
            f"./data/qva_TSCA_result/{group_idx+1}_{group_name}_20.csv", mode="a", newline=""
        ) as file:
            writer = csv.writer(file)
            writer.writerow(row)
