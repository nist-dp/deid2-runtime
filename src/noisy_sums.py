import os
import copy
import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, Normalizer, OneHotEncoder

import sys  
sys.path.insert(0, '../runtime/scripts')
import metric, score

from utils.utils_nn import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scorer = metric.Deid2Metric()

def get_submission(probs, df_ground_truth, df_submission_format):
    sums = df_ground_truth.sum(axis=1).values
    hist = (sums[:, np.newaxis] * probs).astype(int)

    df_submission = df_submission_format.copy()
    epsilons = df_submission.index.levels[0]
    for eps in epsilons:
        df_submission.loc[eps, :] = hist

    return df_submission

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

def naively_add_laplace_noise(arr, scale: float, seed: int = None):
    """
    Add Laplace random noise of the desired scale to the dataframe of counts. Noisy counts will be
    clipped to [0,∞) and rounded to the nearest positive integer.
    Expects a numpy array and a Laplace scale.
    """
    if seed is not None:
        np.random.seed(seed)
    noise = np.random.laplace(scale=scale, size=arr.size).reshape(arr.shape)
    result = np.clip(arr + noise, a_min=0, a_max=np.inf)
    return result.round().astype(np.int)

# import data
df_incidents = pd.read_csv('../data/incidents.csv')

INDEX_COLS = ["epsilon", "neighborhood", "year", "month"]
df_submission_format = pd.read_csv('../data/submission_format.csv', index_col=INDEX_COLS)

df_ground_truth_all = score.get_ground_truth(df_incidents, df_submission_format)
df_ground_truth = df_ground_truth_all.loc[1.0]

sums = df_ground_truth.sum(axis=1).values[:, np.newaxis]

# TODO get max records per user and epsilon from parameters.json

def get_noisy_sums(arr, epsilon: float, c: float, sensitivity: int = 20, seed: int = None):
    scale = sensitivity/(c*epsilon)
    noisy_sums = naively_add_laplace_noise(arr, scale = scale)

    return noisy_sums
