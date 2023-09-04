import pandas as pd

df = pd.read_csv("catpred_100percent_steric.csv")
df2 = pd.read_csv("test.reactant")
source = df2["Reactant"] + "." + df["prediction_1"]
source.to_csv('test.source', index = False, header = False, encoding = 'utf8')

