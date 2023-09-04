import pandas as pd
import numpy as np
import math
import sys
from time import process_time

# Start the stopwatch / counter 
t1_start = process_time() 

# Load fitting results
fitting_results = pd.read_csv('fitting_results_train.csv')

# Load prediction data
input_file = sys.argv[1]
print(f'DFTScore of {input_file} has been calculated')
prediction_data = pd.read_csv(input_file)

#set fitting threshold
threshold = 0.5
print(f'fitting threshold is {threshold}')

# Output file
output_file = input_file[:-4] + '_dft_0p5_train.csv'
with open(output_file, 'w') as output_file:
    # Header
    output_file.write("Reactant,product,er_nbo,er_bond,er_vbur,er_l,er_b1,er_b5,average_er,DFTScore\n")
    
    total_dftscore = 0
    average_dftscore = 0
    count = 0
    # Iterate through prediction data
    for index, row in prediction_data.iterrows():
        # Find corresponding fitting results
        fitting_row = fitting_results[(fitting_results['Reactant'] == row['Reactant']) & (fitting_results['product'] == row['product'])]

        if fitting_row.empty:
            output_file.write(f"{row['Reactant']},{row['product']},not found,not found,not found,not found,not found,not found,not found,not found\n")
            continue

        fitting_row = fitting_row.iloc[0]

        # Compute er values
        er_values = []
        valid_er_values = []
        for column, suffix in zip(['nbo_charge', 'bond_energy', 'Vbur', 'L', 'B1', 'B5'], ['nbo', 'bond', 'vbur', 'l', 'b1', 'b5']):
        #for column, suffix in zip(['bond_energy', 'Vbur', 'L', 'B1', 'B5'], ['bond', 'vbur', 'l', 'b1', 'b5']):
            value = row[column]

            if value in ['N/A', 'nan', 'NaN', 'None']:
                er_values.append('nofit')
                continue

            if value == 'not found':
                er_values.append('0')
                continue

            a = fitting_row[f'a_{suffix}']
            b = fitting_row[f'b_{suffix}']
            r2 = fitting_row[f'r2_{suffix}']
            if a == 'nofit' or b == 'nofit' or r2 == 'nofit':
                er_values.append('nofit')
                continue
            if float(r2) < threshold:
                er_values.append('badfit')
                continue  
            #print(f'a is {a}, b is {b}')
            er_value = float(a) * float(value) + float(b)
            er_values.append(er_value)

        # Compute average er, ignoring 'N/A' values
        for value in er_values:
            if value != 'nofit' and value != 'badfit':
                value = float(value)
                if math.isnan(value):
                    continue
                else:
                    valid_er_values.append(float(value))
        #valid_er_values = np.array([value for value in er_values if value != 'nofit']).astype(np.float)
        #print(valid_er_values)
        if len(valid_er_values) == 0:
            average_er = 'nofit'
            DFTScore = 'nofit'
        else:
            average_er = sum(valid_er_values) / len(valid_er_values)
            #print(f'average_er is {average_er}')
            DFTScore = math.exp(average_er) / (1 + math.exp(average_er))
            #print(f'DFTScore is {DFTScore}')
            if DFTScore != np.nan:
            #else:
                #print(f'valid DFTScore is {DFTScore}')
                total_dftscore += DFTScore
                count += 1
    
        output_file.write(f"{row['Reactant']},{row['product']},{','.join(map(str, er_values))},{average_er},{DFTScore}\n")

average_dftscore = total_dftscore / count
print(f'valid DFTScore count is {count}.')
print(f'average DFTScore is {average_dftscore}.')

# Stop the stopwatch / counter
t1_stop = process_time()

print("Elapsed time:", t1_stop, t1_start) 
   
print("Elapsed time during the whole program in seconds:", t1_stop-t1_start) 
