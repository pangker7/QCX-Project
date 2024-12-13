import numpy as np
import os
import time
import re
import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

# Dictionary mapping elements to their respective bond counts
num_bond = {'H': 1, 'C': 4, 'N': 3, 'O': 2, 'F': 1, 'Si': 4, 'P': 3, 'S': 2, 'Cl': 1, 'K': 1, 'Br': 1, 'I': 1, 'Sn': 4}

def add_suffix_to_elements(V_list0):
    """
    Add suffixes to elements in V_list0 to account for their occurrences.
    """
    count_dict = {}  # Dictionary to track the occurrence count of each element
    V_list = []  # List to store elements with suffixes

    for element in V_list0:
        # Update occurrence count of the element
        count_dict[element] = count_dict.get(element, 0) + 1
        # Append element with its count as suffix
        V_list.append(f"{element}{count_dict[element]}")

    return V_list

def create_tuples(indices, V_list):
    """
    Create tuples of elements from the indices, repeating them based on the third element in the index.
    """
    result = []  # List to store the result tuples

    for triplet in indices:
        # Retrieve elements from V_list using the indices
        first_element = V_list[triplet[0] - 1]  # Adjusting for 0-based indexing
        second_element = V_list[triplet[1] - 1]
        
        # Get repeat count from the third element in the index
        repeat_count = triplet[2]

        # Extend result with repeated tuples
        result.extend([(first_element, second_element)] * repeat_count)

    return result

def parse_mol_file(file_path):
    """
    Parse a MOL file to extract vertex and edge information.
    """
    V0, E0 = 0, 0
    V_list0, E_list0 = [], []

    with open(file_path, 'r') as f:
        lines = f.readlines()
        
        # Locate 'V2000' line and extract V0 and E0 values
        for i, line in enumerate(lines):
            if 'V2000' in line:
                parts = line.split()
                V0 = int(parts[0])  # Number of vertices (V0)
                num_bond_list = np.zeros(V0)
                E0 = int(parts[1])  # Number of edges (E0)

                # Extract vertex elements (4th column) into V_list0
                for j in range(i + 1, i + 1 + V0):
                    V_list0.append(lines[j].split()[3])

                # Extract edge information (first three columns) into E_list0
                for j in range(i + 1 + V0, i + 1 + V0 + E0):
                    e = [int(x) for x in lines[j].split()[:3]]
                    num_bond_list[e[0] - 1] += e[2]
                    num_bond_list[e[1] - 1] += e[2]
                    E_list0.append(e)
                break

        V_list = V_list0.copy()
        E_list = E_list0.copy()
        delta_sum = 0
        flag = True

        for v in range(V0):
            if V_list0[v] != 'H':
                delta = int(num_bond[V_list0[v]] - num_bond_list[v])
                if delta < 0:
                    if V_list[v] == 'P':
                        delta1 = int(5 - num_bond_list[v])
                        if delta1 < 0:
                            print('P, Error in bond count!')
                            flag = False
                        else:
                            V_list += delta1 * ['H']
                            for i in range(delta1):
                                E_list += [[v + 1, V0 + delta_sum + i + 1, 1]]
                            delta_sum += delta1
                    elif V_list[v] == 'S':
                        delta1 = int(6 - num_bond_list[v])
                        if delta1 < 0:
                            print('S, Error in bond count!')
                            flag = False
                        else:
                            V_list += delta1 * ['H']
                            for i in range(delta1):
                                E_list += [[v + 1, V0 + delta_sum + i + 1, 1]]
                            delta_sum += delta1
                    else:
                        print('Error in bond count!')
                        flag = False
                elif delta > 0:
                    V_list += delta * ['H']
                    for i in range(delta):
                        E_list += [[v + 1, V0 + delta_sum + i + 1, 1]]
                delta_sum += delta

    V_list_out = add_suffix_to_elements(V_list)
    E_list_out = create_tuples(E_list, V_list_out)

    return V_list_out, E_list_out, flag

def simplify_chemical_structure(V_list0, E_list0):
    """
    Simplify the chemical structure by removing hydrogen atoms and adjusting the edges.
    """
    # Extract non-hydrogen atoms (vertices)
    non_hydrogen_atoms = [atom for atom in V_list0 if not atom.startswith('H')]

    # Create a dictionary to store the hydrogen bond counts for each non-hydrogen atom
    hydrogen_count = {atom: 0 for atom in non_hydrogen_atoms}

    # Create new edges excluding hydrogen atoms
    new_edges = []
    for edge in E_list0:
        atom1, atom2 = edge
        
        # Count hydrogen bonds
        if atom1.startswith('H'):
            hydrogen_count[atom2] += 1
        elif atom2.startswith('H'):
            hydrogen_count[atom1] += 1
        else:
            # Keep the edge if both atoms are non-hydrogen
            if atom1 in non_hydrogen_atoms and atom2 in non_hydrogen_atoms:
                new_edges.append((atom1, atom2))

    # Generate hydrogen atom connection counts vector (h)
    h = [hydrogen_count[atom] for atom in non_hydrogen_atoms]

    return non_hydrogen_atoms, new_edges, h

def data_collection(file_name):
    """
    Process chemical data from a CSV file, download MOL files, parse and simplify them, and update the CSV.
    """
    # Create a directory for storing the MOL files
    folder_name = 'mol_files'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Set up WebDriver for Chrome
    options = webdriver.ChromeOptions()
    prefs = {"download.default_directory": os.path.abspath(folder_name)}  # Set download directory
    options.add_experimental_option("prefs", prefs)
    driver = webdriver.Chrome(options=options)


    # Read input CSV file
    with open(file_name, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        # Process each row in the CSV file
        for row in reader:
            item = row['CASRN']  # Get CASRN value from the row

            print(f'Processing CASRN: {item}')

            # Open the target website
            driver.get('https://www.chemspider.com/search')

            # Wait for page to load
            time.sleep(3)

            # Find the search bar and input the CASRN
            search_bar = driver.find_element(By.ID, 'input-left-icon')
            search_bar.send_keys(item)
            search_bar.send_keys(Keys.ENTER)

            # Wait for the page to load results
            time.sleep(5)

            # Get the current URL and extract the ChemSpider ID
            current_url = driver.current_url
            match = re.search(r'Chemical-Structure\.(\d+)\.html', current_url)
            if match:
                ChemSpiderID = match.group(1)
                print(f"Spider ID: {ChemSpiderID}")
            else:
                print("No Spider ID found in the URL.")

            # Extract chemical name
            common_tags = ['h3', 'h4', 'h5']
            Name = None
            for tag in common_tags:
                try:
                    name_element = driver.find_element(By.XPATH, f'//div[@class="compound-title"]//{tag}[@id="cmp-title-label"]')
                    Name = name_element.text
                    if Name:
                        print(f"Chemical Name: {Name}")
                        break
                except Exception:
                    continue

            # Extract molecular formula
            try:
                formula_element = driver.find_element(By.XPATH, '//td[@class="molecular-formula"][@id="molecular-fml-value"]')
                formula_html = formula_element.get_attribute('innerHTML')
                formula = re.sub(r'<[^>]+>', '', formula_html)  # Remove HTML tags
                MolecularFormula = formula.replace(' ', '')
                print(f"Molecular Formula: {MolecularFormula}")
            except Exception as e:
                print(f"Error extracting molecular formula: {e}")

            try:
                # Click the download button
                download_button = driver.find_element(By.ID, "button-download")
                ActionChains(driver).move_to_element(download_button).click().perform()

                # Wait for file download to complete
                time.sleep(5)

                print(f"Download complete: {ChemSpiderID}.mol")
                V_list_out0, E_list_out0, flag = parse_mol_file(f'mol_files/{ChemSpiderID}.mol')
                V_list, E_list, h = simplify_chemical_structure(V_list_out0, E_list_out0)

                # If the structure has more than 64 atoms or an error flag is set, delete the file
                if len(V_list) > 64 or not flag:
                    os.remove(f'mol_files/{ChemSpiderID}.mol')
                    print(f"File {ChemSpiderID}.mol deleted, atom count: {len(V_list)} > 64")
                else:
                    # Update the row with extracted data and write it to the new CSV
                    row['Name'] = Name
                    row['MolecularFormula'] = MolecularFormula
                    row['ChemSpiderID'] = ChemSpiderID
                    row['VerticeSet'] = V_list
                    row['EdgeSet'] = E_list
                    row['NumHydrogenSet'] = h

                    # Write updated row to the new CSV file
                    with open('molecule_info.csv', 'a', newline='', encoding='utf-8') as output_csv:
                        writer = csv.DictWriter(output_csv, fieldnames=reader.fieldnames)
                        if output_csv.tell() == 0:  # Write header if file is empty
                            writer.writeheader()
                        writer.writerow(row)
            except Exception as e:
                print(f"Download failed: {ChemSpiderID}.mol - Error: {e}")

    # Close the browser
    driver.quit()

# Start data collection
data_collection('CASRN_list.csv')
