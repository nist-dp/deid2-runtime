import json
import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import sys
sys.path.insert(0, '../runtime/scripts')
import metric, score

INDEX_COLS = ["epsilon", "neighborhood", "year", "month"]
EPS_RUN = {1.0: 0,
           2.0: 1,
           10.0: 2
           }
scorer = metric.Deid2Metric()

def get_data():
    df_incidents = pd.read_csv('../data/incidents.csv')
    df_submission_format = pd.read_csv('../data/submission_format.csv', index_col=INDEX_COLS)
    df_ground_truth = score.get_ground_truth(df_incidents, df_submission_format)
    return df_incidents, df_ground_truth, df_submission_format

def get_incident_sums(df_incidents):
    enc_onehot = OneHotEncoder(sparse=False)
    incidents = np.arange(df_incidents['incident_type'].max()+1)
    enc_onehot.fit(incidents.reshape(-1, 1))
    incidents_encoded = enc_onehot.transform(df_incidents[['incident_type']].values)
    X_incidents = pd.DataFrame(incidents_encoded, columns=incidents)

    X = df_incidents[['sim_resident', 'neighborhood', 'year', 'month']]
    X = pd.concat([X, X_incidents], axis=1)
    X = X.groupby(['sim_resident', 'neighborhood', 'year', 'month']).sum()
    X.reset_index(inplace=True)
    return X, incidents

def get_encoder(df, attr_feat):
    unique_values = []
    for attr in attr_feat:
        unique_values.append(list(np.unique(df[attr])))

    feats = []
    for feat in itertools.product(*unique_values):
        feats.append('_'.join([str(f) for f in feat]))
    feats = np.array(feats)

    enc = LabelEncoder()
    enc.fit(feats)

    return enc, feats

def get_priv_params(epsilon):
    with open('../data/parameters.json') as f:
        parameters = json.load(f)
    runs = parameters["runs"]

    r = runs[EPS_RUN[epsilon]]
    max_epsilon = r["epsilon"]
    delta = r["delta"]
    sensitivity = r["max_records_per_individual"]

    return max_epsilon, delta, sensitivity

def get_score(actual, predicted):
    n_rows, _n_incidents = actual.shape
    raw_penalties = []
    for i in range(n_rows):
        components_i = scorer._penalty_components(actual[i, :], predicted[i, :])
        raw_penalties.append(components_i)
    raw_penalties = np.array(raw_penalties)

    scores = np.ones(raw_penalties.shape[0])
    scores -= raw_penalties.sum(axis=1)
    scores = np.clip(scores, a_min=0.0, a_max=1.0)

    return scores, raw_penalties

def get_submission(probs, df_ground_truth, df_submission_format):
    sums = df_ground_truth.sum(axis=1).values
    hist = (sums[:, np.newaxis] * probs).astype(int)

    df_submission = df_submission_format.copy()
    epsilons = df_submission.index.levels[0]
    for eps in epsilons:
        df_submission.loc[eps, :] = hist

    return df_submission

def naively_add_laplace_noise(arr, scale, seed=None, clip_and_round=True):
    """
    Add Laplace random noise of the desired scale to the dataframe of counts. Noisy counts will be
    clipped to [0,∞) and rounded to the nearest positive integer.
    Expects a numpy array and a Laplace scale.
    """
    if seed is not None:
        np.random.seed(seed)
    if isinstance(scale, np.ndarray):
        assert(scale.shape == arr.shape)
        noise = np.random.laplace(scale=scale)
    else:
        noise = np.random.laplace(scale=scale, size=arr.size).reshape(arr.shape)
    result = arr + noise
    if clip_and_round:
        result = np.clip(result, a_min=0, a_max=np.inf)
        result = result.round().astype(np.int)
    return result

def naively_add_gaussian_noise(arr, scale, seed=None):
    """
    Add Laplace random noise of the desired scale to the dataframe of counts. Noisy counts will be
    clipped to [0,∞) and rounded to the nearest positive integer.
    Expects a numpy array and a Laplace scale.
    """
    if seed is not None:
        np.random.seed(seed)
    if isinstance(scale, np.ndarray):
        noise = np.random.normal(scale=scale)
    else:
        noise = np.random.normal(scale=scale, size=arr.size).reshape(arr.shape)
    result = arr + noise
    result = np.clip(result, a_min=0, a_max=np.inf)
    result = result.round().astype(np.int)
    return result