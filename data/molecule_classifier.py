from pandas._libs.lib import has_infs
from tqdm import tqdm
import csv
import numpy as np
import pandas as pd
import ast
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
include_path = os.path.join(script_dir, "../src")
sys.path.append(include_path)

import preprocess
import basic

def convert_str_to_list_or_tuple(val):
        try:
            return eval(val)
        except:
            return val

def classify_molecules_by_groups(path_groups, path_molecues, path_output):

    df_group = pd.read_csv(path_groups)
    group_strictly_equal = df_group['StrictlyEqual']
    group_name = df_group['GroupName'].apply(convert_str_to_list_or_tuple)
    group_vertice_set = df_group['VerticeSet'].apply(convert_str_to_list_or_tuple)
    group_edge_set = df_group['EdgeSet'].apply(convert_str_to_list_or_tuple)
    group_num_hydrogen_set = df_group['NumHydrogenSet'].apply(convert_str_to_list_or_tuple)
    num_group=len(group_name)

    with open(path_molecues, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        i = 1
        for row in reader:

            molecule_vertice_set = ast.literal_eval(row['VerticeSet'])
            molecule_edge_set = ast.literal_eval(row['EdgeSet'])
            molecule_num_hydrogen_set = ast.literal_eval(row['NumHydrogenSet'])

            print('Molecule: ', i)
            i += 1
            for j in tqdm(range(num_group)):
                mat_A = preprocess.change_to_graph(group_vertice_set[j], group_edge_set[j], group_num_hydrogen_set[j], 1, 1)
                mat_B = preprocess.change_to_graph(molecule_vertice_set, molecule_edge_set, molecule_num_hydrogen_set, 1, 1)


                N = len(mat_A[0,:])
                M = len(mat_B[0,:])
                if N * np.ceil(np.log2(M)) > 20:
                    has_group_value = 2
                else:
                    problem = basic.Problem(mat_A, mat_B, np.array(group_vertice_set[j]), np.array(molecule_vertice_set), same_group_loss=0.2, diff_group_loss=1.0, subgraph=not bool(group_strictly_equal[j]))
                    has_group_value = problem.has_group()

                row[group_name[j]] = int(has_group_value)
            # Write updated row to the new CSV file
            with open(path_output, 'a', newline='', encoding='utf-8') as output_csv:
                writer = csv.DictWriter(output_csv, fieldnames=reader.fieldnames)
                if output_csv.tell() == 0:  # Write header if file is empty
                        writer.writeheader()
                writer.writerow(row)




if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    group_path = os.path.join(script_dir, "fun_group.csv")
    input_path = os.path.join(script_dir, "molecule_info.csv")
    output_path = os.path.join(script_dir, "molecule_info_classified.csv")
    classify_molecules_by_groups(group_path, input_path, output_path)
