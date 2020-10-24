import json
import numpy as np
import pandas as pd
import pdb

attrs = ['neighborhood', 'year', 'month', 'incident_type']

df = pd.read_csv('../data/incidents.csv')
df = df[attrs]
df['year'] = 0
df['month'] -= 1
df.to_csv('./data/incidents.csv', index=False)

with open('../data/parameters.json') as f:
    parameters = json.load(f)
schema = parameters['schema']

domain = {}
for attr in attrs:
    if attr in ['year', 'month']: #TODO: make this more elegant across whole codebase. Define as periods, not year-month
        continue
    domain[attr] = len(schema[attr])

years = np.unique([x['year'] for x in schema['periods']])
domain['year'] = len(years)
months = np.unique([x['month'] for x in schema['periods']])
domain['month'] = len(months)

with open('./data/incidents-domain.json', 'w') as f:
    json.dump(domain, f)