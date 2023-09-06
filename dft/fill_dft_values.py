import csv
import json
import sys
import re

# Load the dictionary from the JSON file
def load_dictionary_from_file(input_file_path):
    with open(input_file_path, 'r') as file:
        return json.load(file)

# Read the smiles1 values from the input CSV file
def read_smiles1_from_csv(input_csv_path):
    smiles1_list = []
    with open(input_csv_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)
        for row in csv_reader:
            smiles1_list.append(row[1])
    return smiles1_list

# Write the data to the output CSV file
def write_data_to_csv(dictionary, smiles1_list, output_csv_path):
    # Define the column headers for the CSV file
    headers = ['smiles1', 'map1', 'nbo_charge', 'bond_energy', 'Vbur', 'smiles2', 'map2', 'L', 'B1', 'B5']

    # Open the CSV file for writing
    with open(output_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write the headers to the CSV file
        csv_writer.writerow(headers)
        # Iterate over the list of smiles1 values
        for smiles1 in smiles1_list:
            # Look up the corresponding values in the dictionary
            data = dictionary.get(smiles1)
            if data:
                # Write the data to the CSV file
                csv_writer.writerow([
                    smiles1,
                    data['map1'],
                    data['nbo_charge'],
                    data['bond_energy'],
                    data['Vbur'],
                    data['smiles2'],
                    data['map2'],
                    data['L'],
                    data['B1'],
                    data['B5']
                    ])
            else:
                csv_writer.writerow([
                    smiles1,
                    'not found',
                    'not found',
                    'not found',
                    'not found',
                    'not found',
                    'not found',
                    'not found',
                    'not found',
                    'not found'
                    ])

# Specify the path to the input JSON file
input_file_path = 'AHO_DFT.json'

# Load the dictionary from the JSON file
dictionary = load_dictionary_from_file(input_file_path)

# Specify the path to the input CSV file containing smiles1 values
input_csv_path = sys.argv[1]

# Read the smiles1 values from the input CSV file
smiles1_list = read_smiles1_from_csv(input_csv_path)

# Specify the path to the output CSV file
output_csv_path = str(input_csv_path[:-4]) + '_filled.csv'

# Write the data to the output CSV file
write_data_to_csv(dictionary, smiles1_list, output_csv_path)                                                                                                                                                                                        
