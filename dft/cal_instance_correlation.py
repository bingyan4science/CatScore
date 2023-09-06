import sys, scipy
from scipy.stats import spearmanr
import random
import pandas as pd
import math
import numpy as np

# Ensure a file name was passed as a command line argument
if len(sys.argv) != 3:
    print("Usage: python script.py <filename dft> <filename cattest>")
    sys.exit(1)

#calcualte Spearman correlation
def cal_corr(array_1, array_2):
    a, b = [], []
    j = 0
    for j in range(len(array_1)):
        if array_1[j] == '':
            continue
        if array_2[j] == '':
            continue
        if array_1[j] == 'nofit' or array_1[j] == 'badfit' or array_1[j] == 'not found':
            continue
        if array_2[j] == 'nofit' or array_2[j] == 'badfit' or array_2[j] == 'not found':
            continue
        a.append(float(array_1[j]) + random.random()*0.001)
        b.append(float(array_2[j]) + random.random()*0.001)

    a = np.array(a)
    b = np.array(b)

    # Calculate Spearman's correlation
    corr, _ = spearmanr(a, b)
    return corr

dft = sys.argv[1]
ftheta = sys.argv[2]

# Load the data from the file specified in the command line argument
df_dft = pd.read_csv(sys.argv[1])
df_ftheta = pd.read_csv(sys.argv[2])

dftscore = df_dft['DFTScore']
accuracy = df_ftheta['rank']
i = 0
for i in range(len(accuracy)):
    if accuracy[i] != 1:
        accuracy[i] = 0
catscore = np.exp(df_ftheta['log_likelihood_total'].astype(float))

corr_dft_accuracy = cal_corr(dftscore, accuracy)
corr_dft_catscore = cal_corr(dftscore, catscore)
print(f'The dft-accuracy Spearman correlation of {ftheta[:-4]} is {corr_dft_accuracy}')
print(f'The dft-catscore Spearman correlation of {ftheta[:-4]} is {corr_dft_catscore}')

df = pd.DataFrame(columns=['dftscore', 'catscore', 'accuracy'], index=range(len(dftscore)))
df['dftscore'] = dftscore
df['accuracy'] = accuracy
df['catscore'] = catscore

df.to_csv(f'{ftheta[:-4]}' + '_dft_accuracy_catscore.csv', index=False)
