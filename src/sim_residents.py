import numpy as np
import pandas as pd

from io import StringIO
from csv import writer 
from tqdm import tqdm

import pdb

df_matched = pd.read_csv('./data/911_data/matched_incidents.csv')
df_matched = df_matched.sort_values(['sim_resident', 'neighborhood', 'year', 'month', 'incident_type'])
df_matched.reset_index(drop=True, inplace=True)

y = df_matched.groupby(['sim_resident']).size()
y = np.bincount(y)
y = y[1:]
y = y / y.sum()

df_incidents = pd.read_csv('./data/raw_incidents.csv')
df_incidents.dropna(inplace=True)
df_incidents.sort_values(['year', 'month', 'day', 'incident_type'], inplace=True)
df_incidents['group'] = df_incidents.groupby(['year', 'lat', 'lon']).ngroup()
df_incidents = df_incidents.sort_values('group')

for year in df_incidents['year'].unique()[::-1]:
    print(year)
    
    mask = df_incidents['year'] == year
    df_incidents_year = df_incidents.loc[mask].reset_index(drop=True)
    print(df_incidents_year.shape)
#     continue
    
    output = StringIO()
    csv_writer = writer(output)
    write_cols = True
    sim_resident = 0

    groups = df_incidents_year['group'].unique()
    for idx, group in enumerate(tqdm(groups)):
        mask = df_incidents_year['group'] == group
        df_group = df_incidents_year[mask].reset_index(drop=True)

        while len(df_group) != 0:
            group_size = np.infty
            if len(df_group) == 1:
                group_size = 0
            while group_size > len(df_group):
                group_size = np.random.choice(np.arange(len(y)), p=y)

            df = df_group.loc[:group_size].reset_index(drop=True)
            df['sim_resident'] = sim_resident
            sim_resident += 1
            del df['group']

            if write_cols:
                columns = df.columns.values
                csv_writer.writerow(columns)
                write_cols = False

            for i in df.index:
                row = df.loc[i].values
                csv_writer.writerow(row)

            df_group = df_group.loc[group_size+1:].reset_index(drop=True)
            
    output.seek(0)
    df = pd.read_csv(output)
    print(df.shape)
    
    df.to_csv('./data/911_data/incidents_{}.csv'.format(year))