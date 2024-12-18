import pandas as pd
import os
import csv
import numpy as np
import matplotlib.pyplot as plt


def convert_str_to_list_or_tuple(val):
    try:
        return eval(val)
    except:
        return val


path_tcsa = "../data/TSCA_result"

group_idx = 2
file_name = [f for f in os.listdir(path_tcsa) if f.startswith(str(group_idx + 1))][0]

df_data = pd.read_csv(f"{path_tcsa}/{file_name}")
len_molecule = df_data["len_molecule"].apply(convert_str_to_list_or_tuple)
valid_prob = df_data["valid_prob"].apply(convert_str_to_list_or_tuple)
sol_prob = df_data["sol_prob"].apply(convert_str_to_list_or_tuple)
d_min = df_data["d_min"].apply(convert_str_to_list_or_tuple)
num_molecule = len(len_molecule)

has_indices = np.where(d_min == 0)[0]
not_has_indices = np.where(d_min != 0)[0]


def get_sorted_arr(prob, indices):
    sorted_dict = {}
    for i in indices:
        x = len_molecule[i]
        if x not in sorted_dict:
            sorted_dict[x] = [prob[i]]
        else:
            sorted_dict[x].append(prob[i])
    x = [k for k, _ in sorted_dict.items()]
    y = [v for _, v in sorted_dict.items()]
    return x, y


def plot_prob(prob, prob_name, indices_list, label_list, color_list, plot_err=True, plot_data=True):
    plt.figure()
    for i, indices in enumerate(indices_list):
        x, y_list = get_sorted_arr(prob, indices)
        avg = [np.mean(data) for data in y_list]
        err = [np.std(data) for data in y_list]
        plt.ylim(0, 1)
        if plot_err:
            plt.errorbar(x, avg, yerr=err, fmt="o", color=color_list[i][1], ecolor="grey", elinewidth=2, capsize=4)
        if plot_data:
            plt.scatter(len_molecule[indices], prob[indices], label=label_list[i], color=color_list[i][0])
    plt.grid()
    plt.legend()
    label_name = "+".join(label_list)
    plot_name = ("plot_err" if plot_err else "") + ("plot_data" if plot_data else "")
    fig_name = prob_name + label_name + plot_name
    plt.savefig(f"{fig_name}.pdf")
    plt.close()


indices_list = [has_indices, not_has_indices]
label_list = ["has", "not_has"]
color_list = [["#1f77b4", "#aec7e8"], ["#ff7f0e", "#ffbb78"]]
plot_prob(valid_prob, "valid", indices_list, label_list, color_list)
plot_prob(valid_prob, "valid", indices_list, label_list, color_list, plot_err=True, plot_data=False)
plot_prob(valid_prob, "valid", [indices_list[0]], [label_list[0]], [color_list[0]], plot_err=True, plot_data=True)
plot_prob(valid_prob, "valid", [indices_list[1]], [label_list[1]], [color_list[1]], plot_err=True, plot_data=True)
plot_prob(sol_prob, "sol", indices_list, label_list, color_list)
plot_prob(sol_prob, "sol", indices_list, label_list, color_list, plot_err=True, plot_data=False)
plot_prob(sol_prob, "sol", [indices_list[0]], [label_list[0]], [color_list[0]], plot_err=True, plot_data=True)
plot_prob(sol_prob, "sol", [indices_list[1]], [label_list[1]], [color_list[1]], plot_err=True, plot_data=True)
# plot_prob(valid_prob, [indices_list[1]], [label_list[1]], [color_list[1]], plot_err=True, plot_data=False)

# plt.scatter(
# len_molecule[not_has_idx], sol_prob[not_has_idx], label="not_has", color="#ff7f0e""#1f77b4"
# )
# plt.title("散点图示例")
# plt.xlabel("X轴")
# plt.ylabel("Y轴")
# plt.show()
