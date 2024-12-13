import preprocess
import basic
import quantum_annealing
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import random
from collections import defaultdict

def read_and_rearrange_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Remove any leading/trailing whitespace from each line
    lines = [line.strip() for line in lines]

    # Group the lines into sets of three
    grouped_data = []
    for i in range(0, len(lines), 3):
        if i + 2 < len(lines):
            group = (lines[i], lines[i + 1], lines[i + 2])
            grouped_data.append(group)

    # Convert each group into a tuple of parsed data
    result = []
    for group in grouped_data:
        # Parse the first line (list of strings)
        first_line = eval(group[0])
        # Parse the second line (list of tuples)
        second_line = eval(group[1])
        # Parse the third line (list of integers)
        third_line = eval(group[2])
        # Combine them into a tuple
        result.append((first_line, second_line, third_line))

    return result

if __name__ == "__main__":
    exp = int(input("No. of experiment. 1 for plotting p - M, 2 for plotting p - m0: "))

    data = read_and_rearrange_data("./molecule.txt")
    random.shuffle(data)

    if exp == 1:
        sol_probs = []
        val_probs = []
        Ms = []
        for cha_b, bonds_b, hydrogen_b in tqdm.tqdm(data):
            cha_a = ['C1', 'O1', 'O2']
            bonds_a = [('C1', 'O1'), ('C1', 'O1'), ('C1', 'O2')]
            hydrogen_a = [0, 0, 1]

            mat_B = preprocess.change_to_graph(cha_b, bonds_b, hydrogen_b, 3, 3)
            mat_A = preprocess.change_to_graph(cha_a, bonds_a, hydrogen_a, 3, 3)

            problem = basic.Problem(mat_A, mat_B, cha_a, cha_b, same_group_loss=0.2, diff_group_loss=1.0)

            if (problem.brutal_force()[0] == 0 and len(problem.brutal_force()[1]) == 1):
                sol = quantum_annealing.QA_simulate(problem, params={'t0': 50, 'm0': 100, 'silent':True})
                sol_probs += [sol['sol_prob']]
                val_probs += [sol['valid_prob']]
                Ms += [sol['problem'].M]

        # plotting
        for data in [sol_probs, val_probs]:
            # Group data by M
            grouped_data = defaultdict(list)
            for M, datum in zip(Ms, data):
                grouped_data[M].append(datum)

            # Prepare data for plotting
            M_values = []
            data_values = []
            average_values = []
            error_values = []

            for M, data_list in grouped_data.items():
                M_values.append(M)
                data_values.extend(data_list)
                if len(data_list) >= 3:
                    average_values.append(np.mean(data_list))
                    error_values.append(np.std(data_list))
                else:
                    average_values.append(None)
                    error_values.append(None)

            # Plot scattered points
            plt.scatter(Ms, data, label='Data Points')

            # Plot average values with error bars
            for M, avg, err in zip(grouped_data.keys(), average_values, error_values):
                if avg is not None and err is not None:
                    plt.errorbar(M, avg, yerr=err, fmt='o', color='red', capsize=5, label='Average with Error' if M == min(grouped_data.keys()) else "")

            # Add labels and title
            plt.xlabel('M')
            plt.ylabel('prob')
            plt.title('Scatter Plot with Average and Error Bars')
            plt.legend()
            plt.show()
            input("Press Enter to continue showing next plot...")
        
        with open("output_find_carboxyl_1.txt", "a") as file:
            file.write(str(Ms) + "\n")
            file.write(str(sol_probs) + "\n")
            file.write(str(val_probs) + "\n")
    elif exp == 2:
        sol_probs = []
        val_probs = []
        m0s = [10, 20, 50, 100, 200, 500]

        for m0 in m0s:
            print(f"Doing {m0}")
            sum_sol_prob = 0
            sum_val_prob = 0
            for cha_b, bonds_b, hydrogen_b in tqdm.tqdm(data):
                cha_a = ['C1', 'O1', 'O2']
                bonds_a = [('C1', 'O1'), ('C1', 'O1'), ('C1', 'O2')]
                hydrogen_a = [0, 0, 1]

                mat_B = preprocess.change_to_graph(cha_b, bonds_b, hydrogen_b, 3, 3)
                mat_A = preprocess.change_to_graph(cha_a, bonds_a, hydrogen_a, 3, 3)

                problem = basic.Problem(mat_A, mat_B, cha_a, cha_b, same_group_loss=0.2, diff_group_loss=1.0)

                # if (problem.brutal_force()[0] == 0 and len(problem.brutal_force()[1]) == 1):
                sol = quantum_annealing.QA_simulate(problem, params={'t0': m0/2, 'm0': m0, 'silent':True})
                sum_sol_prob += sol['sol_prob']
                sum_val_prob += sol['valid_prob']
            sol_probs += [sum_sol_prob / len(data)]
            val_probs += [sum_val_prob / len(data)]

        # plotting
        for data in [sol_probs, val_probs]:
            # Plot scattered points
            plt.scatter(m0s, data, label='Data Points')

            # Add labels and title
            plt.xlabel('m0')
            plt.ylabel('prob')
            plt.title('Scatter Plot')
            plt.legend()
            plt.show()
            input("Press Enter to continue showing next plot...")
        
        with open("output_find_carboxyl_2.txt", "a") as file:
            file.write(str(m0s) + "\n")
            file.write(str(sol_probs) + "\n")
            file.write(str(val_probs) + "\n")