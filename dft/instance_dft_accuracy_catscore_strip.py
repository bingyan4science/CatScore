import csv
import sys, scipy
from scipy.stats import spearmanr
import random
import math
import numpy as np

# Read the input CSV
input_filename = 'instance_dft_accuracy_catscore.csv'
output_filename = 'instance_dft_accuracy_catscore_strip.csv'

data = []
with open(input_filename, 'r') as file:
    reader = csv.reader(file)
    header = next(reader)  # Get the header
    data.append(header)
    
    for row in reader:
        # Check if all cells from the 2nd column onward can be converted to a float
        try:
            [float(cell) for cell in (row[2], row[5], row[8], row[11], row[14], row[17], row[20], row[23], row[26], row[29], row[32], row[35], row[38])]
            data.append(row)
        except ValueError:
            continue

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
        a.append(float(array_1[j]) + random.random()*0.0001)
        b.append(float(array_2[j]) + random.random()*0.0001)

    a = np.array(a)
    b = np.array(b)

    # Calculate Spearman's correlation
    corr, _ = spearmanr(a, b)
    return corr

# Add correlation row and average row to the data
corr_row = ['Correlation', '']
avg_row = ['Average', '']
for col in range(2, len(header)-2, 3):
    dft = [float(row[col]) for row in data[1:]]
    catscore = [float(row[col+1]) for row in data[1:]]
    accuracy = [float(row[col+2]) for row in data[1:]]
    corr_dft_catscore = cal_corr(dft, catscore)
    corr_dft_accuracy = cal_corr(dft, accuracy)
    corr_row.append('')
    corr_row.append(corr_dft_catscore)
    corr_row.append(corr_dft_accuracy)
    avg_dft = sum(dft) / len(dft)
    avg_row.append(avg_dft)
    avg_catscore = sum(catscore) / len(catscore)
    avg_row.append(avg_catscore)
    avg_accuracy = sum(accuracy) / len(accuracy)
    avg_row.append(avg_accuracy)

data.append(corr_row)
data.append(avg_row)

# Write to the output CSV
with open(output_filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)

