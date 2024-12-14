from pandas._libs.lib import has_infs
import csv
import basic
import preprocess
import numpy as np
import pandas as pd
import ast
from tqdm import tqdm

def convert_str_to_list_or_tuple(val):
        try:
            return eval(val)
        except:
            return val

def classify_molecules_by_groups(path_groups, path_molecues, path_output):
    import concurrent.futures
    
    def run_with_timeout(func, timeout):
        # 使用 ThreadPoolExecutor 启动线程
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(func)  # 提交任务
            try:
                result = future.result(timeout=timeout)
                return result  # 返回结果
            except concurrent.futures.TimeoutError:
                print("The function took too long and was skipped.")
                return 2  # 超时后返回 None 或其他你想要的值


    df_group = pd.read_csv(path_groups)
    group_name = df_group['GroupName'].apply(convert_str_to_list_or_tuple)
    group_vertice_set = df_group['VerticeSet'].apply(convert_str_to_list_or_tuple)
    group_edge_set = df_group['EdgeSet'].apply(convert_str_to_list_or_tuple)
    group_num_hydrogen_set = df_group['NumHydrogenSet'].apply(convert_str_to_list_or_tuple)
    num_group=len(group_name)

    df_molecule = pd.read_csv(path_molecues)
    molecule_vertice_set = df_molecule['VerticeSet'].apply(convert_str_to_list_or_tuple)
    molecule_edge_set = df_molecule['EdgeSet'].apply(convert_str_to_list_or_tuple)
    molecule_num_hydrogen_set = df_molecule['NumHydrogenSet'].apply(convert_str_to_list_or_tuple)
    molecule_id = df_molecule['ChemSpiderID'].apply(convert_str_to_list_or_tuple)
    num_molecule=len(molecule_id)


    with open(path_output, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([""] + group_name)

        for i in tqdm(range(num_molecule)):
        # for i in range(num_molecule):
            row = [molecule_id[i]]  # 第一列是分子ID
            for j in range(num_group):
                print(i,j)
                mat_A = preprocess.change_to_graph(group_vertice_set[j], group_edge_set[j], group_num_hydrogen_set[j], 1, 1)
                mat_B = preprocess.change_to_graph(molecule_vertice_set[i], molecule_edge_set[i], molecule_num_hydrogen_set[i], 1, 1)

                problem = basic.Problem(mat_A, mat_B, np.array(group_vertice_set[j]), np.array(molecule_vertice_set[i]), same_group_loss=0.2, diff_group_loss=1.0)
                has_group_value = run_with_timeout(problem.has_group, 0.5)

                row.append(int(has_group_value))

            writer.writerow(row)


if __name__ == '__main__':
    # cha_b = ['C1', 'C2', 'C3', 'C4', 'O1', 'O2']
    # bonds_b = [('C1','C2'), ('C2','C3'), ('C3','C4'),
    #         ('C4','O1'), ('C4','O1'), ('C4','O2')]
    # hydrogen_b = [3, 2, 2, 0, 0, 1]
    # cha_a = ['C1', 'O1', 'O2']
    # bonds_a = [('C1', 'O1'), ('C1', 'O1'), ('C1', 'O2')]
    # hydrogen_a = [0, 0, 1]
    #
    # mat_B = preprocess.change_to_graph(cha_b, bonds_b, hydrogen_b, 3, 3)
    # mat_A = preprocess.change_to_graph(cha_a, bonds_a, hydrogen_a, 3, 3)
    # problem = basic.Problem(mat_A,mat_B,cha_a,cha_b,1,1)
    #
    # print(problem.vec_A)
    # print(problem.vec_B)
    # print("[" + ", ".join([str(row.tolist()) for row in mat_A]) + "]")
    # print("[" + ", ".join([str(row.tolist()) for row in mat_B]) + "]")

    classify_molecules_by_groups("../data/fun_group.csv", "../data/molecule_info.csv", "../data/molecule_group_presence.csv")
