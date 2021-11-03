import pandas as pd

raw_feats = pd.read_csv('raw_results/raw_results.csv', header=None)
raw_feats

for col in raw_feats:
    print(raw_feats[col].unique())