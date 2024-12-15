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

    with open(path_molecues, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        i = 1
        for row in reader:
            if i > 438:
                molecule_vertice_set = ast.literal_eval(row['VerticeSet'])
                molecule_edge_set = ast.literal_eval(row['EdgeSet'])
                molecule_num_hydrogen_set = ast.literal_eval(row['NumHydrogenSet'])

                print('Molecule: ', i)
                i += 1
                # row = [molecule_id[i]]  # 第一列是分子ID
                for j in tqdm(range(num_group)):
                    mat_A = preprocess.change_to_graph(group_vertice_set[j], group_edge_set[j], group_num_hydrogen_set[j], 1, 1)
                    mat_B = preprocess.change_to_graph(molecule_vertice_set, molecule_edge_set, molecule_num_hydrogen_set, 1, 1)


                    N = len(mat_A[0,:])
                    M = len(mat_B[0,:])
                    if N * np.log2(M) > 20:
                        has_group_value = 2
                    else:
                        problem = basic.Problem(mat_A, mat_B, np.array(group_vertice_set[j]), np.array(molecule_vertice_set), same_group_loss=0.2, diff_group_loss=1.0)
                        has_group_value = problem.has_group()

                    row[group_name[j]] = int(has_group_value)
                # Write updated row to the new CSV file
                with open(path_output, 'a', newline='', encoding='utf-8') as output_csv:
                    writer = csv.DictWriter(output_csv, fieldnames=reader.fieldnames)
                    if output_csv.tell() == 0:  # Write header if file is empty
                        writer.writeheader()
                    writer.writerow(row)
            else:
                i += 1



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

    classify_molecules_by_groups("data/fun_group.csv", "data/molecule_group_presence.csv", "data/molecule_group_presence_output.csv")
