import argparse
from scipy import sparse
from scipy.stats import norm, laplace
from scipy.sparse.linalg import lsmr

from utils.utils_data import *
from utils.hdmm import matrix, workload

import sys
sys.path.insert(0, '../runtime/scripts')
import metric, score

import pdb

scorer = metric.Deid2Metric()

def random_hdmm(name, number, marginal, proj=None, seed=0, filter=None):
    data, projections = randomKway(name, number, marginal, proj=proj, seed=seed, filter=filter)
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
    print("l1 = {:.4f}, l2 = {:.4f}".format(l1, l2))
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

def get_args():
    description = ''
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)

    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--workload', type=int, help='queries', default=1)
    parser.add_argument('--workload_seed', type=int, default=0)
    parser.add_argument('--marginal', type=int, help='queries', default=3)
    args = parser.parse_args()

    print(args)
    return args

def get_score(actual, predicted):
    n_rows, _n_incidents = actual.shape
    raw_penalties = []
    for i in range(n_rows):
        components_i = scorer._penalty_components(actual[i, :], predicted[i, :])
        raw_penalties.append(components_i)
    raw_penalties = np.array(raw_penalties)

    scores = np.ones(raw_penalties.shape[0])
    scores -= raw_penalties.sum(axis=1)

    return scores, raw_penalties

if __name__ == '__main__':
    args = get_args()

    data, measurements, workloads = random_hdmm('incidents', args.workload, marginal=args.marginal, seed=args.workload_seed)
    N = data.df.shape[0]

    with open('../data/parameters.json') as f:
        parameters = json.load(f)
    runs = parameters["runs"]

    total_score = 0
    for i, r in enumerate(runs):
        epsilon = r["epsilon"]
        delta = r["delta"]
        sensitivity = r["max_records_per_individual"]

        print("Run {}: epsilon={}".format(i, epsilon))

        answers = run(data, measurements, workloads, eps=epsilon, delta=delta, sensitivity=sensitivity, bounded=False)

        y, proj, W = answers[0]
        data_proj = data.project(proj).datavector()

        ground_truth = data_proj.reshape(data.domain.shape)
        ground_truth = ground_truth.reshape((-1, ground_truth.shape[-1]))

        outputs = y.reshape(data.domain.shape).copy()
        outputs = outputs.reshape((-1, outputs.shape[-1]))
        outputs[outputs < 0] = 0
        scores, penalties = get_score(ground_truth, np.around(outputs))
        print('{:.3f}'.format(scores.sum()))
        print(penalties.mean(axis=0))

        total_score += scores.sum()

    print("TOTAL SCORE: {}".format(total_score))

    # INDEX_COLS = ["epsilon", "neighborhood", "year", "month"]
    # df_incidents = pd.read_csv('../data/incidents.csv')
    # df_submission_format = pd.read_csv('../data/submission_format.csv', index_col=INDEX_COLS)
    # df_ground_truth_all = score.get_ground_truth(df_incidents, df_submission_format)
    # df_ground_truth = df_ground_truth_all.loc[1.0]
    # assert((df_ground_truth.values != ground_truth).sum() == 0)