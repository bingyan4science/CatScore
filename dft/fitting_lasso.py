import pandas as pd
#from sklearn.linear_model import LinearRegression
#from sklearn.metrics import r2_score
import numpy as np
import math
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib

# Read the CSV file
df = pd.read_csv("aho_dataset_train_no_duplicates_prepared.csv")

# Function to calculate er
def calculate_er(ee):
    if ee == 1:
        ee = 0.99999
    return math.log((1 + ee) / (1 - ee))

# Add er column
df['log_er'] = df['ee'].apply(calculate_er)

# Group by Reactant and Product
groups = df.groupby(['Reactant', 'product'])
# List of alpha values to tune
alpha_values = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
# Dictionary to store the best models and results
best_models_and_results = {}
empty = ['N/A', 'nan', 'NaN', 'None', 'not found', '']
descriptors = ['nbo_charge', 'bond_energy', 'Vbur', 'L', 'B1', 'B5']
count_zero = 0

for (reactant, product), group in groups:
    descriptors_filter = ['nbo_charge', 'bond_energy', 'Vbur', 'L', 'B1', 'B5']

    if len(group) >= 3:
        for index, row in group.iterrows():
            #import pdb; pdb.set_trace()
            for descriptor in descriptors:
                if str(row[descriptor]) in empty:
                    if descriptor in descriptors_filter:
                        descriptors_filter.remove(descriptor)
        if len(descriptors_filter) > 0:
            group_name = f'({reactant}, {product})'
            X_data = group[descriptors_filter].values
            Y_data = group['log_er'].values
            # Split the data into training and testing sets
            X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)
            best_mse = float('inf')
            best_model = None
            best_y_pred = None
            best_alpha = None

            # Tune the alpha parameter
            for alpha in alpha_values:
                # Create the Lasso model
                lasso = Lasso(alpha=alpha)
                
                # Fit the model to the training data
                #import pdb; pdb.set_trace()
                try:
                    lasso.fit(X_train, Y_train)
                    #lasso.fit(X_data, Y_data)

                except:
                    import pdb; pdb.set_trace()
                
                # Predict on the test data
                Y_pred = lasso.predict(X_test)
                
                # Evaluate the model
                mse = mean_squared_error(Y_test, Y_pred)
                print(f"Group: {group_name}, Alpha: {alpha}, Mean Squared Error: {mse}")
                
                # Check if this model has the lowest MSE
                #if mse < best_mse and lasso.coef_.sum() != 0:
                if mse < best_mse:
                    best_mse = mse
                    best_model = lasso
                    #best_y_pred = y_pred
                    best_alpha = alpha
            
            # Store the best model and results in the dictionary
            if best_model.coef_.sum() == 0:
                count_zero += 1
            else:
                best_models_and_results[group_name] = {
                    'model': best_model,
                    'alpha': best_alpha,
                    'descriptors': descriptors_filter,
                    'mse': best_mse
                }

# Save the best models and results dictionary to a file
best_models_filename = 'best_lasso_models.joblib'
num_models = len(best_models_and_results)
joblib.dump(best_models_and_results, best_models_filename)
print(f"{num_models} best models and results saved to {best_models_filename}")
print(f'{count_zero} models have zero coefficients')

