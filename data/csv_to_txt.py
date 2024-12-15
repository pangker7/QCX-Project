import csv

def extract_columns_from_csv(input_file, output_file):
    """
    Extract the 'VertexSet', 'EdgeSet', and 'NumHydrogenSet' columns from a CSV file
    and write their content to an output text file.
    """
    with open(input_file, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)  # Read the CSV as a dictionary
        
        # Open the output file in write mode
        with open(output_file, 'w', encoding='utf-8') as outputfile:
            # Iterate through each row in the CSV
            for row in reader:
                # Extract the columns 'VertexSet', 'EdgeSet', and 'NumHydrogenSet'
                vertice_set = row.get('VerticeSet', 'N/A')  # Default to 'N/A' if column not found
                edge_set = row.get('EdgeSet', 'N/A')  # Default to 'N/A' if column not found
                num_hydrogen_set = row.get('NumHydrogenSet', 'N/A')  # Default to 'N/A' if column not found

                # Write to the output file
                outputfile.write(vertice_set + '\n')
                outputfile.write(edge_set + '\n')
                outputfile.write(num_hydrogen_set + '\n')

    print(f"Data has been written to {output_file}")

# Define input and output file paths
input_csv = 'molecule_info.csv'  # Replace with your actual CSV file path
output_txt = 'TSCA_molecule.txt'  # Output file path

# Call the function to process the CSV and write to output.txt
extract_columns_from_csv(input_csv, output_txt)
