import argparse
from utils.utils_general import *
import pdb

def get_df_user_type(df, cols, cols_attr):
    df_user_type = df.groupby(cols).size().to_frame()
    df_user_type.columns = ['num_calls']
    df_user_type.reset_index(cols_attr, inplace=True)
    df_user_type = df_user_type.reset_index().groupby(cols_attr + ['num_calls']).size().to_frame()
    df_user_type.columns = ['count']
    return df_user_type

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, default=2019)

    args = parser.parse_args()
    print(args)
    return args

args = get_args()

cols = ['sim_resident', 'neighborhood', 'year', 'month', 'incident_type']
cols_attr = ['year', 'month', 'neighborhood', 'incident_type']

df_incidents, _, df_submission_format = get_data()
df_incidents = df_incidents[cols].astype(int).sort_values(cols)

year = args.year
df_911 = pd.read_csv('./data/911_data/incidents_{}.csv'.format(year))
df_911 = df_911[cols].astype(int).sort_values(cols)

# so that it works with score.get_ground_truth
df_911['year'] = 2019

# we just ignore the extra incidents that they generated (>159)
incidents = np.arange(df_911.max()['incident_type'] + 1)

df_submission_format = df_submission_format[incidents.astype(str)]
mask = df_911['incident_type'].isin(incidents)
df_911 = df_911[mask].reset_index(drop=True)
mask = df_incidents['incident_type'].isin(incidents)
df_incidents = df_incidents[mask].reset_index(drop=True)

# set public and private datasets
# TODO: this code uses the development dataset as public and 911 data as private (you can swap if you want to test things)
df_private = df_911
df_public = df_incidents

# convert to dataframe of "user types"
df_user_type_private = get_df_user_type(df_private, cols, cols_attr)
df_user_type_public = get_df_user_type(df_public, cols, cols_attr)

# get ground truth for private dataset
df_ground_truth_all = score.get_ground_truth(df_private, df_submission_format)
df_ground_truth = df_ground_truth_all.loc[1.0]

# get queries
"""
The index of dq_queries is the list of queries. Right now it uses the public dataset to generate the list of queries.
We then do a left join with the private dataset to get the answer for every query in df_queries
"""
df_queries = df_user_type_public.copy()

# TODO: add to the index of df_queries (df_queries has a single column = 'count'. You can fill this in with anything (such as 0), since we ignore this column anyway.
# example code of if you wanted to add the queries from the private dataset
df_extra = df_user_type_private.copy()
df_queries = pd.concat([df_queries, df_extra])
# example code of adding a specific query
row_template = df_queries.reset_index().loc[[0]]
row_template.loc[:, ['neighborhood', 'month', 'incident_type', 'num_calls']] = [1, 2, 3, 4]
df_queries.reset_index(inplace=True)
df_queries = pd.concat([df_queries, row_template])
df_queries.set_index(cols_attr + ['num_calls'], inplace=True)

# remove duplicate rows
df_queries.reset_index(inplace=True)
df_queries.drop_duplicates(inplace=True)
df_queries.set_index(cols_attr + ['num_calls'], inplace=True)

# join with private dataset to look up the count for each query
df_queries = pd.merge(df_queries, df_user_type_private, how='left', left_index=True, right_index=True, suffixes=['_public',''])
df_queries = df_queries[['count']]
df_queries.fillna(0, inplace=True)

# print("Naive baseline")
# total_score = 0
# for epsilon in [1.0, 2.0, 10.0]:
#     df_laplace = naively_add_laplace_noise(df_ground_truth, 20 / epsilon)
#     scores, penalties = get_score(df_ground_truth.values, df_laplace.values)
#     score_eps = scores.sum()
#     penalties = np.around(penalties.mean(axis=0), 3)
#     total_score += score_eps
#
#     print("Epsilon {}: {:.3f}\t{}".format(epsilon, score_eps, penalties))
#
# print("Total score: {}".format(total_score))

total_score = 0
for epsilon in [1.0, 2.0, 10.0]:
    df_output = df_queries.copy()
    df_output = naively_add_laplace_noise(df_output, 1 / epsilon)

    df_output = df_output.reset_index()
    df_output['total_count'] = df_output['num_calls'] * df_output['count']
    df_output = pd.pivot_table(df_output, aggfunc='sum',
                                values='total_count', columns=['incident_type'], index=['neighborhood', 'year', 'month'])
    df_output.fillna(0, inplace=True)
    df_output.columns = df_output.columns.values

    # add missing cols
    num_incidents = len(incidents)
    missing_cols = list(set(np.arange(num_incidents)) - set(df_output.columns.values))
    for col in missing_cols:
        df_output.loc[:, col] = 0
    df_output = df_output[np.arange(num_incidents)]

    # add missing rows
    missing_rows_idxs = pd.merge(df_submission_format.loc[1.0], df_output, how='left', left_index=True, right_index=True)
    mask = missing_rows_idxs.isnull().any(axis=1)
    missing_rows_idxs = missing_rows_idxs[mask].reset_index()[['neighborhood', 'year', 'month']].values

    df_output.reset_index(inplace=True)
    row_template = np.zeros(shape=df_output.values[0].shape)
    for idx in missing_rows_idxs:
        row = row_template.copy()
        row[:len(idx)] = idx
        df_output.loc[len(df_output), :] = row

    df_output = df_output.astype(int)
    df_output.sort_values(['neighborhood', 'year', 'month'], inplace=True)
    df_output.set_index(['neighborhood', 'year', 'month'], inplace=True)

    outputs = df_output.values.copy()
    # sums = df_ground_truth.values.sum(axis=1)[:, np.newaxis]
    # probs = np.divide(outputs, sums, out=np.zeros_like(outputs, dtype=float), where=sums != 0)
    # outputs[probs < scorer.threshold] = 0

    scores, penalties = get_score(df_ground_truth.values, outputs)
    score_eps = scores.sum()
    penalties = np.around(penalties.mean(axis=0), 3)
    total_score += score_eps

    print("Epsilon {}: {:.3f}\t{}".format(epsilon, score_eps, penalties))

print("Total score: {}".format(total_score))


###### Ignore
# --------------------
# df = df_user_type_private

# --------------------
# queries = []
# for col in ['year', 'month', 'neighborhood', 'incident_type']:
#     queries.append(df_user_type_public.reset_index()[col].unique())
# queries.append(np.arange(1) + 1) # num_calls
# queries = list(itertools.product(*queries))
# df_extra = pd.DataFrame(queries, columns=cols_attr + ['num_calls'])
# df_extra['count'] = 0
# df = pd.concat([df.reset_index(), df_extra])
# df.sort_values(cols_attr, inplace=True)
# df = df.drop_duplicates()
# df.set_index(cols_attr + ['num_calls'], inplace=True)

# --------------------
# print(df.shape)
#
# from sklearn.preprocessing import OneHotEncoder
# enc = OneHotEncoder(handle_unknown='ignore')
# encoded_cols = enc.fit_transform(df_public[['incident_type', 'month']]).toarray()
#
# from sklearn.decomposition import PCA
# pca_encoded = PCA(n_components=2).fit_transform(encoded_cols)
# pca_encoded = pd.DataFrame(pca_encoded)
# pca_encoded['neighborhood'] = df_public['neighborhood']
# pca_encoded = pca_encoded.groupby('neighborhood').mean()
#
# from sklearn.neighbors import NearestNeighbors
# neigh = NearestNeighbors(n_neighbors=5)
# neigh.fit(pca_encoded)
#
# list_neighbors = []
# num_neighbors = 1
# for i in range(pca_encoded.shape[0]):
#     _, neighbors = neigh.kneighbors(pca_encoded.values[i][np.newaxis, :], num_neighbors + 1)
#     neighbors = neighbors.flatten()[1:]
#     list_neighbors.append(neighbors)
#
# df_extra = []
# for neighborhood, neighbors in enumerate(list_neighbors):
#     mask = df_public['neighborhood'].isin(neighbors)
#     x = df_public.loc[mask, ['month', 'incident_type']].drop_duplicates()
#     x['neighborhood'] = neighborhood
#     x['year'] = 2019
#     x['count'] = 0
#     for num_calls in [1, 2]:
#         x_ = x.copy()
#         x_['num_calls'] = num_calls
#         x_.set_index(cols_attr + ['num_calls'], inplace=True)
#         x_.sort_index(inplace=True)
#         df_extra.append(x_)
# df_extra = pd.concat(df_extra)
# x = pd.concat([df, df_extra])
# x.sort_index(inplace=True)
# x = x.reset_index().drop_duplicates().set_index(cols_attr + ['num_calls'])
# df = x
#
# print(df.shape)