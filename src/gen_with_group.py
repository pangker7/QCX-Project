import random
import networkx as nx

def generate_molecule(M, p, K, group, TRIES=10000):
    """
    Generates a molecule with random branches and cycles, with a group attached to a random spot.
    The total number of atoms is M.
    
    Args:
        M (int): Total number of atoms in the molecule.
        p (float): Average ratio of bonds
        K (int): Amount of groups attached.
        group (int): Type of group. 1 is invalid, 2 for Carbonyl (-CO-), 3 for Carboxyl (-COOH), 4 for Phosphonate (-PO₃H₂), 5 for Imide (-C(O)NHC(O)-), 6 for Benzene.
    
    Returns:
        G (networkx.Graph): The molecule represented as a graph.
    """
    valid = False
    tries = 0
    group_sizes = [0,1,2,4,3,0]
    least_remaining_sizes = [0,1,1,1,2,6]
    M -= group_sizes[group-1]*K
    if M < least_remaining_sizes[group-1]:
        raise ValueError("Molecule must be larger.")
    N = 0
    while valid == False:
        
        G = nx.MultiGraph()
        benzene_groups = []
        
        for i in range(1, M + 1):
            G.add_node(f"C{i}", element="C", hydrogen_count=4)  # Each carbon starts with 4 free bonds
        
        carbons = [node for node in G.nodes if G.nodes[node]["element"] == "C"]
        random.shuffle(carbons)
        # Add a group to random carbon(s)
        for i in range(K):
            if group == 2: #-CO-
                available_carbons = [node for node in carbons if G.nodes[node]["hydrogen_count"] == 4]
                if not available_carbons:
                    raise ValueError("Failed to find appropriate molecule of the given configuration")
                target_carbon = random.choice(available_carbons)
                G.nodes[target_carbon]["hydrogen_count"] -= 2
                G.add_node(f"O{i+1}", element="O", hydrogen_count=0)
                G.add_edge(target_carbon, f"O{i+1}")
                G.add_edge(target_carbon, f"O{i+1}")
            elif group == 3: #-COOH
                available_carbons = [node for node in carbons if G.nodes[node]["hydrogen_count"] == 4]
                if not available_carbons:
                    raise ValueError("Failed to find appropriate molecule of the given configuration")
                target_carbon = random.choice(available_carbons)
                G.nodes[target_carbon]["hydrogen_count"] -= 3
                G.add_node(f"O{2*i+1}", element="O", hydrogen_count=0)
                G.add_node(f"O{2*i+2}", element="O", hydrogen_count=1)
                G.add_edge(target_carbon, f"O{2*i+1}")
                G.add_edge(target_carbon, f"O{2*i+1}")
                G.add_edge(target_carbon, f"O{2*i+2}")
            elif group == 4: #-PO3H2
                available_carbons = [node for node in carbons if G.nodes[node]["hydrogen_count"] == 4]
                if not available_carbons:
                    raise ValueError("Failed to find appropriate molecule of the given configuration")
                target_carbon = random.choice(available_carbons)
                G.nodes[target_carbon]["hydrogen_count"] -= 1
                G.add_node(f"P{i+1}", element="P", hydrogen_count=0)
                G.add_node(f"O{3*i+1}", element="O", hydrogen_count=1)
                G.add_node(f"O{3*i+2}", element="O", hydrogen_count=1)
                G.add_node(f"O{3*i+3}", element="O", hydrogen_count=0)
                G.add_edge(f"P{i+1}",target_carbon)
                G.add_edge(f"O{3*i+1}",f"P{i+1}")
                G.add_edge(f"O{3*i+2}",f"P{i+1}")
                G.add_edge(f"O{3*i+3}",f"P{i+1}")
                G.add_edge(f"O{3*i+3}",f"P{i+1}")
            elif group == 5: #-C(O)NHC(O)-
                available_carbons = [node for node in carbons if G.nodes[node]["hydrogen_count"] == 4]
                if not available_carbons or len(available_carbons) == 1:
                    raise ValueError("Failed to find appropriate molecule of the given configuration")
                target_carbon1, target_carbon2 = random.sample(available_carbons, 2)
                G.nodes[target_carbon1]["hydrogen_count"] -= 3
                G.nodes[target_carbon2]["hydrogen_count"] -= 3
                G.add_node(f"N{i+1}", element="N", hydrogen_count=1)
                G.add_node(f"O{2*i+1}", element="O", hydrogen_count=0)
                G.add_node(f"O{2*i+2}", element="O", hydrogen_count=0)
                G.add_edge(f"O{2*i+1}",target_carbon1)
                G.add_edge(f"O{2*i+1}",target_carbon1)
                G.add_edge(f"O{2*i+2}",target_carbon2)
                G.add_edge(f"O{2*i+2}",target_carbon2)
                G.add_edge(f"N{i+1}",target_carbon1)
                G.add_edge(f"N{i+1}",target_carbon2)
            elif group == 6: #-C=C-C=C-C=C-
                available_carbons = [node for node in carbons if G.nodes[node]["hydrogen_count"] == 4]
                if not available_carbons or len(available_carbons) < 6:
                    raise ValueError("Failed to find appropriate molecule of the given configuration")
                target_carbons = random.sample(available_carbons, 6)
                for carbon in target_carbons:
                    G.nodes[carbon]["hydrogen_count"] -= 3
                G.add_edge(target_carbons[0], target_carbons[1])
                G.add_edge(target_carbons[1], target_carbons[2])
                G.add_edge(target_carbons[1], target_carbons[2])
                G.add_edge(target_carbons[2], target_carbons[3])
                G.add_edge(target_carbons[3], target_carbons[4])
                G.add_edge(target_carbons[3], target_carbons[4])
                G.add_edge(target_carbons[4], target_carbons[5])
                G.add_edge(target_carbons[5], target_carbons[0])
                G.add_edge(target_carbons[5], target_carbons[0])
                benzene_groups += [target_carbons]
            else:
                raise ValueError("Invalid group!!!")
        
        if (N == 0):
            N = sum([G.nodes[node]["hydrogen_count"] for node in G.nodes if G.nodes[node]["element"] == "C"]) // 2
            N = int(N * p)
        else:
            N += 1
        
        # Connect the remaining carbons with random branches and cycles
        carbons = [node for node in G.nodes if G.nodes[node]["element"] == "C"]
        random.shuffle(carbons)
        
        bonds = 0
        while carbons:
            bonds += 1
            if bonds > N:
                break
            available_carbons = [node for node in carbons if G.nodes[node]["hydrogen_count"] > 0]
            if not available_carbons or len(available_carbons) == 1:
                break  # No more available carbons to connect
            
            # Randomly select a carbon to connect to, disable connection inside benzene loop.
            valids = False
            while not valids:
                current_carbon, neighbor_carbon = random.sample(available_carbons, 2)
                valids = True
                for benzene_group in benzene_groups:
                    if current_carbon in benzene_group and neighbor_carbon in benzene_group:
                        valids = False
            G.add_edge(current_carbon, neighbor_carbon)
            G.nodes[current_carbon]["hydrogen_count"] -= 1
            G.nodes[neighbor_carbon]["hydrogen_count"] -= 1
    
        # Ensure the molecule is connected
        if nx.is_connected(G):
            valid = True
        
        tries += 1
        if tries >= TRIES:
            raise ValueError("Failed to find appropriate molecule of the given configuration")
    
    return G


def visualize_molecule(G):
    """
    Visualizes the molecule using networkx and matplotlib.
    
    Args:
        G (networkx.Graph): The molecule represented as a graph.
    """
    import matplotlib.pyplot as plt
    pos = nx.spring_layout(G)

    # Create a dictionary to store edge counts
    edge_counts = {}
    for u, v, key in G.edges(keys=True):
        if (u, v) not in edge_counts:
            edge_counts[(u, v)] = 0
        edge_counts[(u, v)] += 1

    # Create a list of edge widths and colors based on the edge counts
    edge_widths = []
    edge_colors = []
    for u, v in G.edges():
        count = edge_counts.get((u, v), 1)
        edge_widths.append(count)
        edge_colors.append("blue" if count > 1 else "black")

    # Draw the graph
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=2000,
        font_size=10,
        font_weight="bold",
        width=edge_widths,  # Set edge widths
        edge_color=edge_colors,  # Set edge colors
    )

    plt.title("Generated Molecule")
    plt.show()

def molecule_to_text_file(G, filename="molecule.txt"):
    """
    Transforms the molecule into the text file format with three lines:
    1. Atom literals
    2. Bonds
    3. Hydrogen counts
    
    Args:
        G (networkx.Graph): The molecule represented as a graph.
        filename (str): The name of the output text file.
    """
    atoms = [node for node in G.nodes]
    random.shuffle(atoms)
    bond_list = list(G.edges)
    # print(bond_list)
    bonds = []
    for literal in bond_list:
        bonds += [literal[0:2]]
    random.shuffle(bonds)
    
    hydrogen_counts = [G.nodes[node].get("hydrogen_count", 0) for node in atoms]
    
    with open(filename, "a") as file:
        file.write(str(atoms) + "\n")
        file.write(str(bonds) + "\n")
        file.write(str(hydrogen_counts) + "\n")


# Usage
if __name__ == "__main__":
    M = int(input("Enter the total number of atoms (M): "))
    p = float(input("Enter the average ratio of bonds (p): "))
    K = int(input("Enter the total amount of groups you need (K): "))
    group = int(input("Enter name of group. 1 is invalid, 2 for Carbonyl (-CO-), 3 for Carboxyl (-COOH), 4 for Phosphonate (-PO₃H₂), 5 for Imide (-C(O)NHC(O)-), 6 for Benzene: "))
    Num = int(input("Enter the number of molecules you want to create (Num): "))
    for _ in range(Num):
        molecule = generate_molecule(M, p, K, group)
        # visualize_molecule(molecule)
        molecule_to_text_file(molecule)
