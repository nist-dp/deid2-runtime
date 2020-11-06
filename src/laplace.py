import argparse
from utils.utils_general import *
import pdb

def get_df_user_type(df, cols_attr):
    df_user_type = df.groupby(['sim_resident'] + cols_attr).size().to_frame()
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
df_user_type_private = get_df_user_type(df_private, cols_attr)
df_user_type_public = get_df_user_type(df_public, cols_attr)

# get ground truth for private dataset
df_gt_all = score.get_ground_truth(df_private, df_submission_format)
df_gt = df_gt_all.loc[1.0]

# get queries
"""
The index of dq_queries is the list of queries. Right now it uses the public dataset to generate the list of queries.
We then doing a left join with the private dataset to get the answer for every query in df_queries
"""
df_queries = df_user_type_public.copy()
# TODO: add to the index of df_queries (df_queries has a single column = 'count'. You can fill this in with anything (such as 0), since we ignore this column anyway.
# example code of if you wanted to add the queries from the private dataset
"""
df_extra = df_user_type_private.copy()
df_queries = pd.concat([df_queries, df_extra])
"""
# example code of adding a specific query
"""
row_template = df_queries.reset_index().loc[[0]]
row_template.loc[:, ['neighborhood', 'month', 'incident_type', 'num_calls']] = [1, 2, 3, 4]
df_queries.reset_index(inplace=True)
df_queries = pd.concat([df_queries, row_template])
df_queries.set_index(cols_attr + ['num_calls'], inplace=True)
"""

# add adjacent months
df_queries.reset_index(inplace=True)
df_extra = [df_queries]
delta = [1, 2, 3]
delta += [12-d for d in delta]
for x in delta:
    df = df_queries.copy()
    df['month'] -= 1 # scales months to values from 0-11
    df['month'] = (df['month'] + x) % 12
    df['month'] += 1 # changes it back to values 1-12
    df_extra.append(df)

    # testing
    test = np.unique((df['month'] - df_queries['month'] + 12) % 12)
    assert(len(test) == 1)
    assert(test[0] == x)

df_queries = pd.concat(df_extra)
df_queries.set_index(cols_attr + ['num_calls'], inplace=True)

# add extra num_calls
df_queries.reset_index(inplace=True)
mask = (df_queries['count'] > 0) & (df_queries['num_calls'] > 1)
df = df_queries[mask]

df_extra = [df_queries]
for num_calls in np.arange(2) + 1:
    df = df_queries.copy()
    df.loc[:, 'num_calls'] = num_calls
    df_extra.append(df)
df_queries = pd.concat(df_extra)
df_queries.set_index(cols_attr + ['num_calls'], inplace=True)

# remove duplicate queries
df_queries['count'] = 0
df_queries.reset_index(inplace=True)
df_queries.drop_duplicates(inplace=True)
df_queries.set_index(cols_attr + ['num_calls'], inplace=True)
df_queries.sort_index(inplace=True)

# join with private dataset to look up the count for each query
df_queries = pd.merge(df_queries, df_user_type_private, how='left', left_index=True, right_index=True, suffixes=['_public',''])
df_queries = df_queries[['count']]
df_queries.fillna(0, inplace=True)

print(df_queries.shape)
print((df_queries['count'] > 0).mean())
print(df_queries['count'].sum())

# run DP mechanism
outputs = score.get_ground_truth(df_public, df_submission_format)
outputs = outputs.loc[1.0]
print("\nRelease public dataset as private (just to get a sense of how good it is)")
scores, penalties = get_score(df_gt.values, outputs.values)
total_score = 3 * scores.sum()
print("Total score: {}".format(total_score))

print("\nLaplace benchmark")
total_score = 0
for epsilon in [1.0, 2.0, 10.0]:
    df_laplace = naively_add_laplace_noise(df_gt, 20 / epsilon)
    scores, penalties = get_score(df_gt.values, df_laplace.values)
    score_eps = scores.sum()
    penalties = np.around(penalties.mean(axis=0), 3)
    total_score += score_eps

    print("Epsilon {}: {:.3f}\t{}".format(epsilon, score_eps, penalties))
print("Total score: {}".format(total_score))

print("\nOur method")
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

    df_output.sort_values(['neighborhood', 'year', 'month'], inplace=True)
    df_output.set_index(['neighborhood', 'year', 'month'], inplace=True)

    # outputs = df_output.values.copy()
    # sums = df_ground_truth.values.sum(axis=1)[:, np.newaxis]
    # probs = np.divide(outputs, sums, out=np.zeros_like(outputs, dtype=float), where=sums != 0)
    # outputs[(probs < scorer.threshold + 0.02) & (probs > scorer.threshold - 0.02)] = 0

    outputs = df_output.values.copy()
    scores, penalties = get_score(df_gt.values, outputs)
    score_eps = scores.sum()
    penalties = np.around(penalties.mean(axis=0), 3)
    print("Epsilon {}: {:.3f}\t{}".format(epsilon, score_eps, penalties))

    total_score += score_eps

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

# --------------------
# from sklearn.metrics import mutual_info_score
# def calc_MI(x, y, bins):
#     c_xy = np.histogram2d(x, y, bins)[0]
#     mi = mutual_info_score(None, None, contingency=c_xy)
#     return mi
#
# pdb.set_trace()
#
# x = score.get_ground_truth(df_public, df_submission_format)
# x = x.loc[1.0]
# x = x.copy().values
# bins = 5
# n = x.shape[1]
# matMI = np.zeros((n, n))
# for i in np.arange(n):
#     for j in np.arange(i+1, n):
#         matMI[i, j] = calc_MI(x[:, i], x[:, j], bins)
# incidents_mi_sorted = np.dstack(np.unravel_index(np.argsort(matMI.ravel()), (n, n)))[0][::-1]
#
# print(df_queries.shape)
# df_queries.reset_index(inplace=True)
#
# df_extra = [df_queries]
# for i in [1]:
#     incidents_mi = incidents_mi_sorted[i]
#     mask = df_queries['incident_type'].isin(incidents_mi)
#     df = df_queries[mask]
#     for i in incidents_mi:
#         for num_calls in np.arange(3) + 1:
#             df.loc[:, 'incident_type'] = i
#             df.loc[:, 'num_calls'] = num_calls
#             df_extra.append(df.copy())
# df_queries = pd.concat(df_extra)
# df_queries.set_index(cols_attr + ['num_calls'], inplace=True)

# -------
# df_gt_public = score.get_ground_truth(df_public, df_submission_format)
# df_gt_public = df_gt_public.loc[1.0]
# df_gt_public_norm = df_gt_public / df_gt_public.sum()
# df_gt_public_norm.reset_index(inplace=True)
# df_gt_public_norm = pd.melt(df_gt_public_norm, id_vars=['neighborhood', 'year', 'month'], value_vars=incidents.astype(str))
# df_gt_public_norm.columns = ['neighborhood', 'year', 'month', 'incident_type', 'frac']
# df_gt_public_norm['incident_type'] = df_gt_public_norm['incident_type'].astype(int)
#
# mask = df_gt_public_norm['frac'] > 1e-5
# df_gt_public_norm = df_gt_public_norm[mask]
# df_queries.reset_index(inplace=True)
# df_queries = pd.merge(df_queries, df_gt_public_norm, left_on=['neighborhood', 'year', 'month', 'incident_type'], right_on=['neighborhood', 'year', 'month', 'incident_type'])
# del df_queries['frac']
# df_queries.set_index(cols_attr + ['num_calls'], inplace=True)
# df_queries.sort_index(inplace=True)
