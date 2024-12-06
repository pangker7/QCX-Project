import numpy as np

def get_atom_type(c):
    j = 0
    while not ('0' <= c[j] <= '9'):
        j += 1
    return c[0:j]

def register_atoms(cha_atom): # temporary solution
    unique_list = []
    for c in cha_atom:
        if c not in unique_list: unique_list.append(c)
    
    l = len(unique_list)

    embedding = {}
    i = 0

    for c in unique_list:
        embedding[c] = 1.0 * i / l
        i += 1
    
    return embedding


# for example:
# cha: [C1,C2,O1,O2]
# bonds: [(C1,C2),(C2,O1),(C2,O1),(C2,O2)]
# hydrogen: [3,0,0,1]
# output: graph adjacency matrix
# diagonals are: H-1.0 O-0.6 C-0.3
# off-diagonal terms are: n for n existence of (A,B) or (B,A). matrix is symmetric.
def change_to_graph(cha, bonds, hydrogen, max_adj, max_hydrogen):
    n = len(cha)
    adj_matrix = np.zeros((n, n))
    hydrogen = np.array(hydrogen)
    
    for bond in bonds:
        i, j = bond
        i = cha.index(i)
        j = cha.index(j)
        adj_matrix[i, j] += 1
        adj_matrix[j, i] += 1
    
    # normalize
    adj_matrix = adj_matrix / max_adj
    hydrogen = hydrogen / max_hydrogen

    for i, hydrogen_num in enumerate(hydrogen):
        adj_matrix[i, i] = hydrogen_num
    
    return adj_matrix

if __name__ == "__main__":
    # Example usage:
    cha = ['C1','C2','O1','O2']
    bonds = [('C1','C2'),('C2','O1'),('C2','O1'),('C2','O2')]
    hydrogen = [3,0,0,1]

    adj_matrix = change_to_graph(cha, bonds, hydrogen, 3, 3)
    print(adj_matrix)