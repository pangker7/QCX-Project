import numpy as np

# This file should be named as "chemistry"

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
def change_to_graph(cha, bonds, hydrogen, max_adj=3, max_hydrogen=3):
    n = len(cha)
    adj_matrix = np.zeros((n, n))
    hydrogen = np.array(hydrogen)
    
    if len(bonds) == 0:
        adj_matrix[0][0]=hydrogen[0]
        return adj_matrix
    
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

# THIS IS A GLOBAL ATOM LIST!
atom_list = [
["Li","Na","K","Rb","Cs"],
["Be","Mg","Ca","Sr","Ba"],
["B","Al","Ga","In","Tl"],
["C","Si","Ge","Sn","Pb"],
["N","P","As","Sb","Bi"],
["O","S","Se","Te"],
["F","Cl","Br","I"]
]
# ATOM LIST END!

class CharacterListQuery:
    def __init__(self, list_of_lists):
        self.char_to_indices = {}
        for index, sublist in enumerate(list_of_lists):
            for char in sublist:
                if char not in self.char_to_indices:
                    self.char_to_indices[char] = set()
                self.char_to_indices[char].add(index)

    def query(self, lit_1, lit_2, same_group_loss, diff_group_loss):
        if lit_1 == lit_2:
            return 0

        if lit_1 not in self.char_to_indices or lit_2 not in self.char_to_indices:
            return diff_group_loss # unknown element is very different from any known ones!
        
        common_indices = self.char_to_indices[lit_1] & self.char_to_indices[lit_2]
        if common_indices:
            return same_group_loss
        
        return diff_group_loss

def get_list_query():
    return CharacterListQuery(atom_list)
