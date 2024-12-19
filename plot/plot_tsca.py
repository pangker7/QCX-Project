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


path_tcsa = "qva_TSCA_result"
# path_tcsa = "../data/qva_TSCA_result"

group_idx = 16
df_group = pd.read_csv("../data/fun_group.csv")
group_name_list = df_group["GroupName"].apply(convert_str_to_list_or_tuple)
file_name = str(group_idx + 1) + f"_{group_name_list[group_idx]}" + "_20"
# file_name = [f for f in os.listdir(path_tcsa) if f.startswith(str(group_idx + 1))][0]

df_data = pd.read_csv(f"../data/{path_tcsa}/{file_name}" + ".csv")
len_molecule = df_data["len_molecule"].apply(convert_str_to_list_or_tuple)
valid_prob = df_data["valid_prob"].apply(convert_str_to_list_or_tuple)
sol_prob = df_data["sol_prob"].apply(convert_str_to_list_or_tuple)
d_min = df_data["d_min"].apply(convert_str_to_list_or_tuple)
degeneracy = df_data["degeneracy"].apply(convert_str_to_list_or_tuple)
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
    plt.scatter(len_molecule[indices], prob[indices], label=label, color=color, s=3, marker='o')
    # plt.scatter(len_molecule[indices], prob[indices], label=label, color=color, s=12, marker='x')
    # plt.scatter(len_molecule[indices], prob[indices], label=label, color=color, s=5, facecolors='none', edgecolors=color)



def plot_err(prob, indices, color, ecolor, markersize=4.5):
    x, y_list = get_sorted_arr(prob, indices)
    avg = [np.mean(data) for data in y_list]
    err = [np.std(data) for data in y_list]
    # plt.errorbar(x, avg, yerr=err, fmt="o", color=color, ecolor=ecolor, elinewidth=1, capsize=3, markersize=markersize)
    plt.errorbar(x, avg, fmt="o", color=color, ecolor=ecolor, elinewidth=1, capsize=3, markersize=markersize)

def plot_avg(prob, indices, label, color):
    x, y_list = get_sorted_arr(prob, indices)
    avg = [np.mean(data) for data in y_list]
    plt.plot(x, avg, label=label, color=color, linestyle='--')


def plot_degeneracy_data(prob, indices, label, ylim=(0, 1)):
    plt.figure()
    plt.ylim(ylim)
    color_list = ["#cccceb", "#8d8dd2", "#8080cd", "#4d4d7b", "#333352", "#26263e", "#1a1a29", "#0d0d15", "#000000"]
    lenth = degeneracy[indices]
    dict ={}
    for i in indices:
        dege = degeneracy[i]
        if dege not in dict:
            dict[dege] = [[len_molecule[i]],[prob[i]]]
        else:
            dict[dege][0].append(len_molecule[i])
            dict[dege][1].append(prob[i])
    for dege, data in dict.items():
        plt.scatter(data[0], data[1], color=color_list[dege], s=300)
    plot_err(prob, indices, "red", ecolor="grey", markersize=20)
    plt.grid()
    plt.legend()
    plt.show()
    

# plot_degeneracy_data(sol_prob, has_indices, "has")

def plot_total_fig(prob, prob_name, indices_list, label_list, color_list, ecolor, plot_err_flag=True, plot_data_flag=True, ylim=(0, 1)):
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

def plot_sub_fig(prob, prob_name, indices, label, color_list, ecolor, ylim=(0, 1)):
    plt.figure()
    plt.ylim(ylim)
    i = 0 if label == "has" else 1
    plot_data(prob, indices, "data points", color_list[i])
    # plot_err(prob, indices, color_list[2], ecolor)
    plot_avg(prob, indices, f"Average p_{prob_name}", color_list[2])
    # plt.grid()
    plt.xlabel("Number of atoms (excluding H) in a molecule")
    plt.ylabel(f"p_{prob_name}")
    plt.legend()
    plt.savefig(f"../figure/{path_tcsa}/{file_name}_{label}_{prob_name}.pdf")
    plt.close()



indices_list = [has_indices, not_has_indices]
label_list = ["has", "not_has"]
# [data, err]
# color_list = [["#1f77b4", "#aec7e8"], ["#ff7f0e", "#ffbb78"]]
color_list = ["#1f70a9", "black", "#c22f2f"]

# plot_total_fig(sol_prob, "sol", indices_list=indices_list, label_list=label_list, color_list=color_list, ecolor="grey", ylim=(0, 1))
# plot_total_fig(valid_prob, "valid", indices_list=indices_list, label_list=label_list, color_list=color_list, ecolor="grey", ylim=(0.7, 1))
plot_sub_fig(valid_prob, "valid", has_indices, "has", color_list, ecolor="grey", ylim=(0.0, 1))
plot_sub_fig(valid_prob, "valid", not_has_indices, "not_has", color_list, ecolor="grey", ylim=(0.0, 1))
plot_sub_fig(sol_prob, "sol", has_indices, "has", color_list, ecolor="grey")
plot_sub_fig(sol_prob, "sol", not_has_indices, "not_has", color_list, ecolor="grey")