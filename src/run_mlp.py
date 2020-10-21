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

# import data
df_incidents = pd.read_csv('../data/incidents.csv')

INDEX_COLS = ["epsilon", "neighborhood", "year", "month"]
df_submission_format = pd.read_csv('../data/submission_format.csv', index_col=INDEX_COLS)

df_ground_truth_all = score.get_ground_truth(df_incidents, df_submission_format)
df_ground_truth = df_ground_truth_all.loc[1.0]

# preprocess data
enc = LabelEncoder()
v_neighborhoods = np.unique(df_incidents['neighborhood'])
v_months = np.unique(df_incidents['month'])
feats = []
for n, m in itertools.product(v_neighborhoods, v_months):
    feats.append('{}_{}'.format(n, m))
feats = np.array(feats)
enc.fit(feats)

df_incidents['feat'] = df_incidents['neighborhood'].astype(str) + '_' + df_incidents['month'].astype(str)
y = df_incidents['incident_type'].values
X = df_incidents['feat'].values
X = enc.transform(X)[:, np.newaxis]
residents = df_incidents['sim_resident'].values
resident_to_idxs = {}
for i, resident in enumerate(residents):
    if resident in resident_to_idxs.keys():
        resident_to_idxs[resident].append(i)
    else:
        resident_to_idxs[resident] = [i]
sampler = UserSampler(resident_to_idxs)
# sampler = None

# embed_dim = 1024
num_embeds = enc.classes_.shape[0]
embed_dim = num_embeds * 1
embedders = []
embedder = nn.Embedding(num_embeds, embed_dim).to(device)
embedders.append(embedder)

num_classes = y.max() - y.min() + 1
batch_size = 514
num_epochs = 500
criterion = nn.CrossEntropyLoss()

dataset = SimpleDataset(X, y)
if sampler is None:
    dataloader_train = DataLoader(dataset, batch_size=batch_size, shuffle=True)
else:
    dataloader_train = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    num_epochs *= 5

X_test = enc.transform(feats)[:, np.newaxis]
dataset_test = SimpleDataset(X_test, y[:len(X_test)]) # y doesn't matter
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

# define model
num_layers = 2
model = Net_embedder(embedders, [embed_dim // 2] * num_layers + [1024], num_classes)
model.to(device)

optimizer = torch.optim.RMSprop(model.parameters(), 1e-4)

feats_idx = [x.split('_') for x in feats]
feats_idx = [[int(x[0]), int(x[1])] for x in feats_idx]
feats_idx = np.array(feats_idx)

df_submission = pd.DataFrame()
df_submission['neighborhood'] = feats_idx[:, 0]
df_submission['mood'] = feats_idx[:, 1]
for i in range(num_classes):
    df_submission[i] = np.zeros(len(df_submission))
df_submission.set_index(['neighborhood', 'mood'], inplace=True)
sums = df_ground_truth.sum(axis=1).values[:, np.newaxis]

best_loss = np.infty
best_model = None

# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs * len(dataloader_train))

# NON-PRIVATE
for i in range(num_epochs):
    # loss = train(model, dataloader_train, optimizer, criterion, device, scheduler=scheduler)
    loss = train(model, dataloader_train, optimizer, criterion, device, scheduler=None)
    scheduler.step(loss)

    print('{}: {:.4f}'.format(i, loss))

    if loss < best_loss:
        best_loss = loss
        best_model = copy.deepcopy(model)

        _, outputs = evaluate(model, dataloader_test, criterion, device)
        probs = nn.functional.softmax(outputs, dim=1)
        probs = probs.clone().detach().numpy()
        outputs = (probs * sums).astype(int)
        outputs[outputs < 0] = 0
        df_submission[np.arange(num_classes)] = outputs
        scores, penalties = get_score(df_ground_truth.values, df_submission.values)
        print('{:.3f}'.format(scores.sum()))
        print(penalties.mean(axis=0))

        outputs[probs < scorer.threshold] = 0
        df_submission[np.arange(num_classes)] = outputs
        scores, penalties = get_score(df_ground_truth.values, df_submission.values)
        print('{:.3f}'.format(scores.sum()))
        print(penalties.mean(axis=0))

    # if epsilon > max_epsilon:
    #     break
    #

#
# # PRIVATE
# import json
# with open('../data/parameters.json') as f:
#     parameters = json.load(f)
# runs = parameters["runs"]
#
# r = runs[-1]
# max_epsilon = r["epsilon"]
# delta = r["delta"]
# sensitivity = r["max_records_per_individual"]
#
# from opacus import PrivacyEngine
# privacy_engine = PrivacyEngine(model,
#                                batch_size=batch_size,
#                                sample_size=len(sampler),
#                                alphas=range(2, 32),
#                                noise_multiplier=1.1,
#                                max_grad_norm=1,
#                                target_delta=delta)
# privacy_engine.attach(optimizer)
#
# # initial result
# for i in range(num_epochs):
#     loss = train(model, dataloader_train, optimizer, criterion, device, scheduler=scheduler)
#     print('{}: {:.4f}'.format(i, loss))
#
#     epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(delta)
#     print(
#         f"Train Epoch: {i} \t"
#         f"Loss: {np.mean(loss):.6f} "
#         f"(ε = {epsilon:.2f}, δ = {delta}) for α = {best_alpha}")