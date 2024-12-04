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
# cha: [C1,H1,H2,H3,C2,O1,O2,H4]
# bonds: [(C1,H1),(C1,H2),(C1,H3),(C1,C2),(C1,O1),(C1,O1),(C1,O2),(O2,H4)]
# output: graph adjacency matrix
# diagonals are: H-1.0 O-0.6 C-0.3
# off-diagonal terms are: n for n existence of (A,B) or (B,A). matrix is symmetric.
def change_to_graph(cha, bonds, embedding=None):
    cha_atom = [get_atom_type(c) for c in cha]
    if embedding is None:
        embedding = register_atoms(cha_atom)
    
    n = len(cha)
    adj_matrix = np.zeros((n, n))
    
    for bond in bonds:
        i, j = bond
        i = cha.index(i)
        j = cha.index(j)
        adj_matrix[i, j] += 1
        adj_matrix[j, i] += 1
    
    # normalize
    adj_matrix = adj_matrix / np.max(adj_matrix)
    
    for i in range(n):
        adj_matrix[i, i] = embedding[cha_atom[i]]
    
    return embedding, adj_matrix

if __name__ == "__main__":
    # Example usage:
    cha = ['C1', 'H1', 'H2', 'H3', 'C2', 'O1', 'O2', 'H4']
    bonds = [('C1', 'H1'), ('C1', 'H2'), ('C1', 'H3'), ('C1', 'C2'), ('C1', 'O1'), ('C1', 'O1'), ('C1', 'O2'), ('O2', 'H4')]

    _, adj_matrix = change_to_graph(cha, bonds)
    print(adj_matrix)