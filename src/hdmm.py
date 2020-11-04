import argparse
from scipy import sparse
from scipy.stats import norm, laplace
from scipy.sparse.linalg import lsmr
from utils.utils_general import *
from utils.utils_data import *
from utils.hdmm import matrix, workload
import pdb

def randomKwayData(data, number, marginal, seed=0, check_size=True):
    prng = np.random.RandomState(seed)
    total = data.df.shape[0]
    dom = data.domain
    if check_size:
        proj = [p for p in itertools.combinations(data.domain.attrs, marginal) if dom.size(p) <= total]
    else:
        proj = [p for p in itertools.combinations(data.domain.attrs, marginal)]
    if len(proj) > number:
        proj = [proj[i] for i in prng.choice(len(proj), number, replace=False)]
    return proj

def random_hdmm(data, number, marginal):
    projections = randomKwayData(data, number, marginal, seed=0, check_size=False)
    lookup = {}
    lookup_W = {}

    A100 = sparse.csr_matrix(np.load('utils/hdmm/prefix-100.npy'))
    A101 = sparse.csr_matrix(np.load('utils/hdmm/prefix-100-missing.npy'))
    P100 = workload.Prefix(100)
    P101 = workload.Prefix(101)

    for attr in data.domain:
        n = data.domain.size(attr)
        if n == 100:
            lookup[attr] = A100
            lookup_W[attr] = P100
        elif n == 101:
            lookup[attr] = A101
            lookup_W[attr] = P101
        else:
            lookup[attr] = sparse.eye(n, format='csr')
            lookup_W[attr] = matrix.Identity(n)

    measurements = []
    workloads = []

    for proj in projections:
        Q = reduce(sparse.kron, [lookup[a] for a in proj]).tocsr()
        measurements.append((proj, Q))
        W = matrix.Kronecker([lookup_W[a] for a in proj])
        workloads.append((proj, W))

    return data, measurements, workloads

def run(dataset, measurements, workloads,  eps=1.0, delta=0.0, sensitivity=1.0, bounded=True, seed=None, query_mask=None):
    """
    Run a mechanism that measures the given measurements and runs inference.
    This is a convenience method for running end-to-end experiments.
    """
    state = np.random.RandomState(seed)
    l1 = 0
    l2 = 0
    for _, Q in measurements:
        if query_mask is not None: # doesn't seem to actually matter
            Q = Q[query_mask, :] # there's definitely a cleaner way for indexing this
            Q = Q[:, query_mask]
        l1 += np.abs(Q).sum(axis=0).max()
        try:
            l2 += Q.power(2).sum(axis=0).max()  # for spares matrices
        except:
            l2 += np.square(Q).sum(axis=0).max()  # for dense matrices

    l1 *= sensitivity
    l2 *= sensitivity
    # print("l1 = {:.4f}, l2 = {:.4f}".format(l1, l2))
    if bounded:
        total = dataset.df.shape[0]
        l1 *= 2
        l2 *= 2

    if delta > 0:
        noise = norm(loc=0, scale=np.sqrt(l2 * 2 * np.log(2 / delta)) / eps)
    else:
        noise = laplace(loc=0, scale=l1 / eps)

    x_bar_answers = []
    local_ls = {}
    for proj, A in measurements:
        x = dataset.project(proj).datavector()
        z = noise.rvs(size=A.shape[0], random_state=state)
        a = A.dot(x)
        y = a + z
        if query_mask is not None:
            y = y[query_mask]
            A = A[query_mask, :]
            A = A[:, query_mask]
        local_ls[proj] = lsmr(A, y)[0]

    answers = []
    for proj, W in workloads:
        answers.append((local_ls[proj], proj, W))

    return answers

def get_outputs(counts, df_template):
    df_template['count'] = counts
    df_template['total_count'] = df_template['num_calls'] * df_template['count']

    df = pd.pivot_table(df_template, aggfunc='sum', values='total_count', columns=['incident_type'], index=['neighborhood', 'year', 'month'])
    df.fillna(0, inplace=True)

    # missing cols
    missing_incidents = list(set(incidents) - set(df.columns.values))
    for incident in missing_incidents:
        df[incident] = 0
    df = df[incidents]
    df.columns = incidents

    # missing rows
    missing_rows_idxs = pd.merge(df_submission_format.loc[1.0], df, how='left', left_index=True, right_index=True)
    mask = missing_rows_idxs.isnull().any(axis=1)
    missing_rows_idxs = missing_rows_idxs[mask].reset_index()[['neighborhood', 'year', 'month']].values
    df.reset_index(inplace=True)
    row_template = np.zeros(shape=df.values[0].shape)
    for idx in missing_rows_idxs:
        row = row_template.copy()
        row[:len(idx)] = idx
        df.loc[len(df), :] = row
    df = df.astype(int)
    df.set_index(['neighborhood', 'year', 'month'], inplace=True)
    df.sort_index(inplace=True)

    return df

def get_df_user_type(df, cols, cols_attr):
    df_user_type = df.groupby(cols).size().to_frame()
    df_user_type.columns = ['num_calls']
    df_user_type.reset_index(cols_attr, inplace=True)
    return df_user_type

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, default=2019)

    args = parser.parse_args()
    print(args)
    return args

args = get_args()

with open('../data/parameters.json') as f:
    parameters = json.load(f)
runs = parameters["runs"]

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
_df_private = df_911
_df_public = df_incidents

df_submission_format = df_submission_format[incidents.astype(str)]
df_ground_truth_all = score.get_ground_truth(_df_private, df_submission_format)
df_ground_truth = df_ground_truth_all.loc[1.0]

# convert to dataframe of "user types"
df_private = get_df_user_type(_df_private, cols, cols_attr).reset_index(drop=True)
df_public = get_df_user_type(df_incidents, cols, cols_attr).reset_index(drop=True)

for df in [df_private, df_public]:
    df['year'] -= 2019
    df['month'] -= 1

attrs = cols_attr + ['num_calls']
domain = {}
for attr in attrs:
    domain[attr] = df_private[attr].max() + 1
domain['num_calls'] = 20
domain = Domain(domain.keys(), domain.values())
data = Dataset(df_private, domain)
data, measurements, workloads = random_hdmm(data, 1, marginal=len(domain))

# set up template
df_template = df_public
# df_template = df_private
df_template = df_template.groupby(list(domain.config.keys())).size().to_frame()
df_template.columns = ['count']
df_template.reset_index(inplace=True)
df_template.sort_values(cols_attr + ['num_calls'], inplace=True)

# add adjacent months
# df_extra = [df_template]
# delta = [1, 2]
# delta += [12-d for d in delta]
# for x in delta:
#     df = df_template.copy()
#     df['month'] = (df['month'] + x) % 12
#     df_extra.append(df)
#
#     # testing
#     test = np.unique((df['month'] - df_template['month'] + 12) % 12)
#     assert(len(test) == 1)
#     assert(test[0] == x)
#
# df_template = pd.concat(df_extra)
# df_template['count'] = 0
# df_template.drop_duplicates(inplace=True)
# df_template.sort_values(cols_attr + ['num_calls'], inplace=True)
# df_template.reset_index(drop=True, inplace=True)

query_idxs = []
for col in list(domain.config.keys()):
    query_idxs.append(list(df_template[col].values))
query_idxs = tuple(query_idxs)

num_queries = np.prod(data.domain.shape)
query_mask = np.zeros(num_queries).astype(bool)
x = np.arange(num_queries).reshape(data.domain.shape)[query_idxs]
query_mask[x] = True

df_template['year'] += 2019
df_template['month'] += 1

# checking some things
# answers = run(data, measurements, workloads, eps=1, delta=0, sensitivity=1, query_mask=query_mask)
# y, proj, W = answers[0]
# counts = data.project(proj).datavector()[query_mask]
# df_gt_hdmm = get_outputs(counts, df_template)
# assert((df_gt_hdmm.values == df_ground_truth.values).mean() == 1)

# run HDMM
total_laplace_score = 0
total_gaussian_score = 0
for r in runs:
    epsilon = r["epsilon"]
    delta = r["delta"]

    # Laplace noise
    answers = run(data, measurements, workloads, eps=epsilon, delta=0, sensitivity=1, bounded=False, query_mask=query_mask)
    y, proj, W = answers[0]

    counts = y.copy()
    counts[counts < 0] = 0
    counts = np.around(counts)
    df_outputs = get_outputs(counts, df_template)
    scores, penalties = get_score(df_ground_truth.values, df_outputs.values)
    total_laplace_score += scores.sum()

    # Gaussian noise
    answers = run(data, measurements, workloads, eps=epsilon, delta=delta, sensitivity=1, bounded=False, query_mask=query_mask)
    y, proj, W = answers[0]

    counts = y.copy()
    counts[counts < 0] = 0
    counts = np.around(counts)
    df_outputs = get_outputs(counts, df_template)
    scores, penalties = get_score(df_ground_truth.values, df_outputs.values)
    total_gaussian_score += scores.sum()

print('Laplace: {:.3f}'.format(total_laplace_score))
print('Gaussian: {:.3f}'.format(total_gaussian_score))
