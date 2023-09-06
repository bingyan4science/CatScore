import sys
import pandas as pd
import math
import numpy as np

# Ensure a file name was passed as a command line argument
if len(sys.argv) != 2:
    print("Usage: python script.py <filename dft>")
    sys.exit(1)

ftheta = sys.argv[1]

# Load the data from the file specified in the command line argument
df_ftheta = pd.read_csv(sys.argv[1])

accuracy = df_ftheta['rank']
i = 0
for i in range(len(accuracy)):
    if accuracy[i] != 1:
        accuracy[i] = 0
catscore = np.exp(df_ftheta['log_likelihood_total'].astype(float))

df = pd.DataFrame(columns=['catscore', 'accuracy'], index=range(len(catscore)))
df['accuracy'] = accuracy
df['catscore'] = catscore

df.to_csv(f'{ftheta[:-4]}' + '_catscore_accuracy.csv', index=False)

