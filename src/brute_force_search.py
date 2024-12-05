import itertools
import numpy as np


# get subgraph of H given vertices
def extract_subgraph(adj_H, vertices):
    n = len(vertices)
    subgraph = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            subgraph[i][j] = adj_H[vertices[i]][vertices[j]]
    return subgraph


# find closest subgraph of H compare to G given distance function
def find_closest_subgraph(adj_G, adj_H, distance_function):
    n, m = len(adj_G), len(adj_H)
    min_distance = float("inf")
    best_subgraph = None
    best_vertices = None

    # 枚举 H 的所有大小为 n 的顶点子集
    for vertices in itertools.combinations(range(m), n):
        # 提取子图的邻接矩阵
        adj_F = extract_subgraph(adj_H, vertices)
        # 计算距离
        dist = distance_function(adj_G, adj_F)
        # 更新最优解
        if dist < min_distance:
            min_distance = dist
            best_subgraph = adj_F
            best_vertices = vertices

    return best_vertices, min_distance


# find minimum weight given injection f
def find_min_weight(N, M, weight_function):
    if M < N:
        raise ValueError("M cannot be larger than N!")

    min_weight = float("inf")
    best_f = None

    # 枚举所有大小为 N 的排列
    for perm in itertools.permutations(range(M), N):
        # perm 是一个可能的 f
        weight = weight_function(perm)
        if weight < min_weight:
            min_weight = weight
            best_f = perm

    return best_f, min_weight


if __name__ == "__main__":

    def distance_function(adj_G, adj_F):
        return np.linalg.norm(adj_G - adj_F, ord="fro")

    adj_G = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])

    adj_H = np.array([[0, 1, 0, 0], [1, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0]])

    vertices, min_distance = find_closest_subgraph(adj_G, adj_H, distance_function)
    if vertices is not None:
        print(f"找到最接近的子图，顶点集为: {vertices}")
        print(f"最小距离为: {min_distance}")
    else:
        print("未找到有效子图")
