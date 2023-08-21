import sys, scipy
from scipy.stats import spearmanr
import numpy as np
import random
import csv

dft_selectivity = 'aug14/selectivity_dft.csv'

a, b, c, d = [], [], [], []

with open(sys.argv[1]) as fin:
    reader = csv.reader(fin)
    next(reader)
    for row in reader:
        loglikelihood = float(row[7])
        #print(loglikelihood)
        a.append(np.exp(loglikelihood))

with open(dft_selectivity) as fin_dft:
    reader = csv.reader(fin_dft)
    for row in reader:
        b.append(row[0])

for i in range(len(b)):
    if b[i] == 'nofit' or b[i] == 'badfit':
        continue
    if b[i] == 'not found':
        continue
    c.append(float(a[i]) + random.random()*0.0001)
    d.append(float(b[i]) + random.random()*0.0001)

c = np.array(c)
d = np.array(d)

# Calculate Spearman's correlation
corr, _ = spearmanr(c, d)
print('Spearmans correlation a and b: %.3f' % corr)
#print ('mse', ((a-b)*(a-b)).mean())
