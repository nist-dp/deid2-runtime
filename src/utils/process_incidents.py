import json
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import pdb

df = pd.read_csv('../data/incidents.csv')
pdb.set_trace()
attrs = ['neighborhood', 'month', 'incident_type']
df = df[attrs]
df['month'] -= 1

enc = LabelEncoder()
domain = {}
for col in attrs:
    domain[col] = int(df[col].max() + 1)

df.to_csv('./data/incidents.csv', index=False)

with open('./data/incidents-domain.json', 'w') as f:
    json.dump(domain, f)