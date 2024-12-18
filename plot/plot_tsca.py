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


def plot_data(prob, indices, label, color):
    plt.scatter(len_molecule[indices], prob[indices], label=label, color=color)


def plot_err(prob, indices, color, ecolor):
    x, y_list = get_sorted_arr(prob, indices)
    avg = [np.mean(data) for data in y_list]
    err = [np.std(data) for data in y_list]
    plt.errorbar(x, avg, yerr=err, fmt="o", color=color, ecolor=ecolor, elinewidth=2, capsize=4)


def plot_fig(prob, prob_name, indices_list, label_list, color_list, ecolor, plot_err_flag=True, plot_data_flag=True, ylim=(0, 1)):
    """
    indices_list = [has, not_has], label_list = [has, not_has], color_list = [has, not_has, avg]
    """
    plt.figure()
    plt.ylim(ylim)
    if plot_data_flag:
        for i in range(2):
            plot_data(prob, indices_list[i], label_list[i], color_list[i])
    if plot_err_flag:
        plot_err(prob, range(num_molecule), color_list[2], ecolor)
    plt.grid()
    plt.legend()
    label_name = "+".join(label_list)
    plot_name = ("err" if plot_err else "") + ("data" if plot_data else "")
    fig_name = prob_name + label_name + plot_name
    plt.savefig(f"{fig_name}.pdf")
    plt.close()


indices_list = [has_indices, not_has_indices]
label_list = ["has", "not_has"]
# [data, err]
# color_list = [["#1f77b4", "#aec7e8"], ["#ff7f0e", "#ffbb78"]]
color_list = ["#1f77b4", "#ff7f0e", "red"]

plot_fig(valid_prob, "valid", indices_list=indices_list, label_list=label_list, color_list=color_list, ecolor="grey", ylim=(0.7, 1))
