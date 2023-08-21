import sys, scipy
from scipy.stats import spearmanr
import numpy as np
import random
import csv

experiment_selectivity = 'aug14/selectivity_exp.csv'

a, b = [], []

with open(sys.argv[1]) as fin:
    reader = csv.reader(fin)
    next(reader)
    for row in reader:
        loglikelihood = float(row[7])
        #print(loglikelihood)
        a.append(np.exp(loglikelihood))

with open(experiment_selectivity) as fin_exp:
    reader = csv.reader(fin_exp)
    for row in reader:
        b.append(float(row[0]))

a = np.array(a)
b = np.array(b)

# Calculate Spearman's correlation
corr, _ = spearmanr(a, b)
print('Spearmans correlation a and b: %.3f' % corr)
print ('mse', ((a-b)*(a-b)).mean())
