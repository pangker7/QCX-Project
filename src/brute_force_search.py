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

    # enumerate all vertices set
    for vertices in itertools.combinations(range(m), n):
        adj_F = extract_subgraph(adj_H, vertices)
        dist = distance_function(adj_G, adj_F)
        if dist < min_distance:
            min_distance = dist
            best_subgraph = adj_F
            best_vertices = vertices

    return best_vertices, min_distance


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
