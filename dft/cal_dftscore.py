import pandas as pd
import numpy as np
import math
import sys
from time import process_time
import joblib
from sklearn.linear_model import Lasso

# Start the stopwatch / counter 
t1_start = process_time() 


# To load the best models and results later
# Load the best models and results dictionary
best_models_filename = 'best_lasso_models.joblib'
loaded_best_models_and_results = joblib.load(best_models_filename)
print("Best models and results loaded")

# Load prediction data
input_file = sys.argv[1]
print(f'DFTScore of {input_file} has been calculated')
prediction_data = pd.read_csv(input_file)

empty = ['N/A', 'nan', 'NaN', 'None', 'not found', '']
descriptors = ['nbo_charge', 'bond_energy', 'Vbur', 'L', 'B1', 'B5']
total_dftscore = 0
average_dftscore = 0
count = 0

# Iterate through prediction data
# Output file
output_file = input_file[:-4] + '_dft_lasso.csv'
with open(output_file, 'w') as output_file:
    for index, row in prediction_data.iterrows():
        #import pdb; pdb.set_trace()
        reactant = row['Reactant']
        product = row['product']
        group_name_to_load = f'({reactant}, {product})'
        if group_name_to_load in loaded_best_models_and_results:
            loaded_group_data = loaded_best_models_and_results[group_name_to_load]
            loaded_model = loaded_group_data['model']
            loaded_descriptors = loaded_group_data['descriptors']
            loaded_alpha = loaded_group_data['alpha']
            for descriptor in loaded_descriptors:
                if str(row[descriptor]) in empty:
                    row[descriptor] = 0
            #import pdb; pdb.set_trace()
            X_test = row[loaded_descriptors].values
            X_test = X_test.reshape(1, -1)
            Y_pred = loaded_model.predict(X_test)
            DFTScore = math.exp(Y_pred) / (1 + math.exp(Y_pred))
            count += 1
            total_dftscore += DFTScore
            output_file.write(f"{reactant},{product},{loaded_alpha},{Y_pred},{DFTScore}\n")
        else:
            output_file.write(f"{reactant},{product},not_found,not_found,not_found\n")
    
average_dftscore = total_dftscore / count
print(f'valid DFTScore count is {count}')
print(f'average DFTScore is {average_dftscore}')

# Stop the stopwatch / counter
t1_stop = process_time()
print("Elapsed time:", t1_stop, t1_start) 
print("Elapsed time during the whole program in seconds:", t1_stop-t1_start) 
