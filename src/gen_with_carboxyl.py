import random
import networkx as nx

def generate_molecule(M, N):
    """
    Generates a molecule with random branches and cycles, with a carboxyl group attached to a random leaf carbon.
    The total number of carbon atoms is M.
    
    Args:
        M (int): Total number of carbon atoms in the molecule.
        N (float): Average num of bonds of target.
    
    Returns:
        G (networkx.Graph): The molecule represented as a graph.
        carboxyl_carbon (str): The label of the carbon atom with the carboxyl group.
    """
    valid = False
    while valid == False:
        if M < 2:
            raise ValueError("M must be at least 2 to form a valid molecule.")
        
        G = nx.MultiGraph()
        
        for i in range(1, M + 1):
            G.add_node(f"C{i}", element="C", hydrogen_count=4)  # Each carbon starts with 4 free bonds
        
        # Add a carboxyl group to a random carbon
        carboxyl_carbon = f"C{random.randint(1, M)}"
        G.nodes[carboxyl_carbon]["hydrogen_count"] -= 3
        G.add_node("O1", element="O", hydrogen_count=0)
        G.add_node("O2", element="O", hydrogen_count=1)
        G.add_edge(carboxyl_carbon, "O1")
        G.add_edge(carboxyl_carbon, "O1")
        G.add_edge(carboxyl_carbon, "O2")
        
        # Connect the remaining carbons with random branches and cycles
        carbons = [node for node in G.nodes if G.nodes[node]["element"] == "C"]
        random.shuffle(carbons)
        
        while carbons:
            if random.random() < 1/N:
                break
            available_carbons = [node for node in carbons if G.nodes[node]["hydrogen_count"] > 0]
            if not available_carbons or len(available_carbons) == 1:
                break  # No more available carbons to connect
            
            # Randomly select a carbon to connect to
            current_carbon = random.choice(available_carbons)
            neighbor_carbon = random.choice(available_carbons)
            if current_carbon == neighbor_carbon:
                continue
            G.add_edge(current_carbon, neighbor_carbon)
            G.nodes[current_carbon]["hydrogen_count"] -= 1
            G.nodes[neighbor_carbon]["hydrogen_count"] -= 1
    
        # Ensure the molecule is connected
        if nx.is_connected(G):
            valid = True
    
    return G, carboxyl_carbon


def visualize_molecule(G):
    """
    Visualizes the molecule using networkx and matplotlib.
    
    Args:
        G (networkx.Graph): The molecule represented as a graph.
    """
    import matplotlib.pyplot as plt
    
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=2000, font_size=10, font_weight="bold")
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
    bond_list = list(G.edges)
    # print(bond_list)
    bonds = []
    for literal in bond_list:
        bonds += [literal[0:2]]
    
    hydrogen_counts = [G.nodes[node].get("hydrogen_count", 0) for node in G.nodes]
    
    with open(filename, "a") as file:
        file.write(str(atoms) + "\n")
        file.write(str(bonds) + "\n")
        file.write(str(hydrogen_counts) + "\n")


# Usage
if __name__ == "__main__":
    M = int(input("Enter the total number of carbon atoms (M): "))
    N = float(input("Enter the average number of bonds (N): "))
    Num = int(input("Enter the number of molecules you want to create (Num): "))
    for _ in range(Num):
        molecule, _ = generate_molecule(M, N)
        # visualize_molecule(molecule)
        molecule_to_text_file(molecule)