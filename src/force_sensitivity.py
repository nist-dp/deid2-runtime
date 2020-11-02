from scipy import sparse
from scipy.stats import norm, laplace
from scipy.sparse.linalg import lsmr
from utils.utils_data import *
from utils.utils_general import *
from utils.hdmm import matrix, workload
import pdb

# Following functions are specifically for running HDMM
def randomKwayData(data, number, marginal, seed=0):
    prng = np.random.RandomState(seed)
    total = data.df.shape[0]
    dom = data.domain
    proj = [p for p in itertools.combinations(data.domain.attrs, marginal)]# if dom.size(p) <= total]
    if len(proj) > number:
        proj = [proj[i] for i in prng.choice(len(proj), number, replace=False)]
    return proj

def randomKway(df, name, number, marginal, seed=0, proj=None):
    domain = "data/{}-domain.json".format(name)
    config = json.load(open(domain))
    domain = Domain(config.keys(), config.values())

    data = Dataset(df, domain)
    if proj is not None:
        data = data.project(proj)
    return data, randomKwayData(data, number, marginal, seed)

def random_hdmm(df, name, number, marginal, proj=None):
    data, projections = randomKway(df, name, number, marginal, proj=proj)
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

def run(dataset, measurements, workloads,  eps=1.0, delta=0.0, sensitivity=1.0, bounded=True, seed=None):
    """
    Run a mechanism that measures the given measurements and runs inference.
    This is a convenience method for running end-to-end experiments.
    """
    state = np.random.RandomState(seed)
    l1 = 0
    l2 = 0
    for _, Q in measurements:
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
        local_ls[proj] = lsmr(A, y)[0]

    answers = []
    for proj, W in workloads:
        answers.append((local_ls[proj], proj, W))

    return answers

# get data
df_incidents, df_ground_truth_all, df_submission_format = get_data()
df_incidents.reset_index(drop=True)

df_ground_truth = df_ground_truth_all.loc[1.0]

attrs = ['incident_type', 'neighborhood', 'year', 'month']
df_incidents['feat'] = df_incidents[attrs[0]].astype(str)
for i in range(1, len(attrs)):
    df_incidents['feat'] = df_incidents['feat'] + '_' + df_incidents[attrs[i]].astype(str)

df_counts = df_incidents.groupby(['sim_resident', 'feat']).size().to_frame()
df_counts.columns = ['count']

df_sensitivites = df_counts.reset_index()[['sim_resident', 'count']].groupby(['sim_resident']).max()
df_sensitivites.columns = ['sensitivity']
sensitivites = df_sensitivites['sensitivity'].unique()
sensitivites.sort()

df_incidents = pd.merge(df_incidents, df_sensitivites, left_on='sim_resident', right_index=True, how='left')
df_incidents.reset_index(drop=True)

with open('../data/parameters.json') as f:
    parameters = json.load(f)
runs = parameters["runs"]

df = df_incidents.copy()
df['year'] = 0
df['month'] -= 1
hdmm_proj = ['neighborhood', 'year', 'month', 'incident_type']
data, measurements, workloads = random_hdmm(df, 'incidents', 1, marginal=4, proj=hdmm_proj)
answers = run(data, measurements, workloads, eps=1, delta=0, sensitivity=1)
y, proj, W = answers[0]
ground_truth_hdmm = data.project(proj).datavector(flatten=False)
ground_truth_hdmm = ground_truth_hdmm.reshape((-1, ground_truth_hdmm.shape[-1]))

assert((ground_truth_hdmm == df_ground_truth.values).mean() == 1)

results = []
df_incidents_shuffled = df_incidents.sample(frac=1)
for s in np.arange(20)+1:
    df = df_incidents_shuffled.groupby(['sim_resident', 'feat']).head(s).reset_index(drop=True)
    frac_kept = df.shape[0] / df_incidents.shape[0]

    df_gt = score.get_ground_truth(df, df_submission_format).loc[1.0]
    scores, penalties = get_score(df_ground_truth.values, df_gt.values)
    sampling_score = scores.sum() * len(runs)

    df_s = df.groupby(attrs + ['sim_resident']).size().to_frame()
    df_s.columns = ['sensitivity']
    df_s = df_s.reset_index().groupby(attrs).max()[['sensitivity']]
    df_s.reset_index(inplace=True)
    df_s = pd.pivot_table(df_s, values='sensitivity', columns=['incident_type'], index=['neighborhood', 'year', 'month'])
    df_s.fillna(0, inplace=True)
    df_s = pd.merge(df_gt.reset_index()[['neighborhood', 'year', 'month']], df_s.reset_index(), how='left',
                    left_on=['neighborhood', 'year', 'month'], right_on=['neighborhood', 'year', 'month'])
    df_s.set_index(['neighborhood', 'year', 'month'], inplace=True)
    missing = set(np.arange(174)) - set(df_s.columns.values)
    for i in missing:
        df_s[i] = 0
    df_s = df_s[np.arange(174)]
    df_s.fillna(0, inplace=True)
    df_s[df_s == 0] = 0

    # for HDMM
    df['year'] = 0
    df['month'] -= 1

    print("Sensitivity Threshold={}, Fraction={:.8f}".format(s, frac_kept))
    laplace_score, gaussian_score, hdmm_score, hdmm0_score = 0, 0, 0, 0
    for r in runs:
        epsilon = r["epsilon"]
        delta = r["delta"]

        # Laplace Mechanism
        scale = df_s.values / epsilon
        outputs = naively_add_laplace_noise(df_gt.values, scale=scale)
        scores, penalties = get_score(df_ground_truth.values, outputs)
        laplace_score += scores.sum()

        # Gaussian Mechanism
        scale = np.sqrt(df_s.values * 2 * np.log(2 / delta)) / epsilon
        df_private = naively_add_gaussian_noise(df_gt, scale=scale)
        outputs = df_private.values
        scores, penalties = get_score(df_ground_truth.values, outputs)
        gaussian_score += scores.sum()

    results.append([s,
                    round(frac_kept, 8),
                    round(sampling_score, 3),
                    round(laplace_score, 3),
                    round(gaussian_score, 3),
                    ])

    print('Sampling: {:.3f}'.format(sampling_score))
    print('Laplace: {:.3f}'.format(laplace_score))
    print('Gaussian: {:.3f}'.format(gaussian_score))
    print('\n')

result_cols = ['sensitivity', 'frac_kept', 'sampling', 'laplace', 'gaussian']
df_results = pd.DataFrame(results, columns=result_cols)

#############

results = []
df_incidents_shuffled = df_incidents.sample(frac=1)
for s in np.arange(20)+1:
    df = df_incidents_shuffled.groupby(['sim_resident', 'feat']).head(s).reset_index(drop=True)
    frac_kept = df.shape[0] / df_incidents.shape[0]

    df_gt = score.get_ground_truth(df, df_submission_format).loc[1.0]
    scores, penalties = get_score(df_ground_truth.values, df_gt.values)
    sampling_score = scores.sum() * len(runs)

    # for HDMM
    df['year'] = 0
    df['month'] -= 1

    print("Sensitivity Threshold={}, Fraction={:.8f}".format(s, frac_kept))
    laplace_score, gaussian_score, hdmm_score, hdmm0_score = 0, 0, 0, 0
    for r in runs:
        epsilon = r["epsilon"]
        delta = r["delta"]

        # Laplace Mechanism
        scale = s / epsilon
        df_private = naively_add_laplace_noise(df_gt, scale=scale)
        outputs = df_private.values
        scores, penalties = get_score(df_ground_truth.values, outputs)
        laplace_score += scores.sum()
        # print('{:.3f}'.format(scores.sum()))
        # print(penalties.mean(axis=0))

        # Gaussian Mechanism
        scale = np.sqrt(s * 2 * np.log(2 / delta)) / epsilon
        df_private = naively_add_gaussian_noise(df_gt, scale=scale)
        outputs = df_private.values
        scores, penalties = get_score(df_ground_truth.values, outputs)
        gaussian_score += scores.sum()

        # HDMM
        data, measurements, workloads = random_hdmm(df, 'incidents', 1, marginal=4, proj=hdmm_proj)
        answers = run(data, measurements, workloads, eps=epsilon, delta=delta, sensitivity=s, bounded=False)

        y, proj, W = answers[0]
        outputs = y.reshape(data.domain.shape).copy()
        outputs = outputs.reshape((-1, outputs.shape[-1]))
        outputs[outputs < 0] = 0
        outputs = np.around(outputs)
        scores, penalties = get_score(ground_truth_hdmm, outputs)
        hdmm_score += scores.sum()

        # HDMM - delta=0
        data, measurements, workloads = random_hdmm(df, 'incidents', 1, marginal=4, proj=hdmm_proj)
        answers = run(data, measurements, workloads, eps=epsilon, delta=0, sensitivity=s, bounded=False)

        y, proj, W = answers[0]
        outputs = y.reshape(data.domain.shape).copy()
        outputs = outputs.reshape((-1, outputs.shape[-1]))
        outputs[outputs < 0] = 0
        outputs = np.around(outputs)
        scores, penalties = get_score(ground_truth_hdmm, outputs)
        hdmm0_score += scores.sum()

    results.append([s,
                    round(frac_kept, 8),
                    round(sampling_score, 3),
                    round(laplace_score, 3),
                    round(gaussian_score, 3),
                    round(hdmm_score, 3),
                    round(hdmm0_score, 3)
                    ])

    print('Sampling: {:.3f}'.format(sampling_score))
    print('Laplace: {:.3f}'.format(laplace_score))
    print('Gaussian: {:.3f}'.format(gaussian_score))
    print('HDMM: {:.3f}'.format(hdmm_score))
    print('HDMM (delta=0): {:.3f}'.format(hdmm0_score))
    print('\n')

result_cols = ['sensitivity', 'frac_kept', 'sampling', 'laplace', 'gaussian', 'hdmm', 'hdmm0']
df_results = pd.DataFrame(results, columns=result_cols)


