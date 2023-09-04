import csv

def swap_symbols(string):
    # Use a placeholder for "@@" first
    string = string.replace("@@", "<PLACEHOLDER>")
    # Then replace "@" with "@@"
    string = string.replace("@", "@@")
    # Finally, replace the placeholder with "@"
    string = string.replace("<PLACEHOLDER>", "@")
    return string

input_file = 'aho_dataset_train.csv'
output_source_file = 'train.source'
output_target_file = 'train.target'

with open(input_file, 'r') as csvfile, open(output_source_file, 'w') as source_file, open(output_target_file, 'w') as target_file:
    reader = csv.reader(csvfile)
    next(reader)  # Skip header row
    count = 0
    for row in reader:
        product_smiles = row[2]

        if '@' in product_smiles:
            reactant_smiles = row[1]
            count += 1
            if '@' in reactant_smiles:
                print(f'Warning! Line {count*2-1} reactant {reactant_smiles} has steric center.')
            
            catalyst_smiles = row[7]
            source_line = f'{reactant_smiles}.{catalyst_smiles}\n'
            source_file.write(source_line)
            source_file.write(source_line)  # Write the line twice
            
            ee = float(row[14])
            target_line_1 = f'{product_smiles},{(1 + ee) / 2}\n'
            target_file.write(target_line_1)
            
            modified_product_smiles = swap_symbols(product_smiles)
                
            target_line_2 = f'{modified_product_smiles},{(1 - ee) / 2}\n'
            target_file.write(target_line_2)

