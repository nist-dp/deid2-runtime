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

# softmax = torch.exp(logits / T) / torch.exp(logits / T).sum(axis=1, keepdims=True)

def KLDivLoss(P, Q, T=1):
    kld = nn.KLDivLoss(size_average=False)
    P_soft = F.softmax(P)
    Q_soft = F.softmax(Q)
    return kld(P_soft.log(), Q_soft)

def SoftCrossEntropyLoss(logits, target):
    logprobs = F.log_softmax(logits, dim=1)
    loss = -(target * logprobs).sum() / logits.shape[0]
    return loss

def L2Norm(input, target):
    return torch.norm(input - target, p=2)

def L1Norm(input, target):
    return torch.norm(input - target, p=1)

def LPNorm(logits, target, alpha=1):
    probs = F.softmax(logits, dim=1)
    loss = 0
    loss += alpha * torch.norm(probs - target, p=2) * 100
    loss += (1 - alpha) * torch.norm(probs - target, p=1)
    return loss

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

# df_ground_truth_normalized = df_ground_truth / df_ground_truth.sum(axis=1)[:, np.newaxis]
# df_ground_truth_normalized.fillna(0, inplace=True)

# import data
df_incidents = pd.read_csv('../data/incidents.csv')

INDEX_COLS = ["epsilon", "neighborhood", "year", "month"]
df_submission_format = pd.read_csv('../data/submission_format.csv', index_col=INDEX_COLS)

df_ground_truth_all = score.get_ground_truth(df_incidents, df_submission_format)
df_ground_truth = df_ground_truth_all.loc[1.0]

X = df_incidents[['sim_resident', 'neighborhood', 'month', 'incident_type']]
enc_onehot = OneHotEncoder(sparse=False)
incidents = np.arange(X['incident_type'].max()+1)
enc_onehot.fit(incidents.reshape(-1, 1))
incidents_encoded = enc_onehot.transform(X[['incident_type']].values)
X_incidents = pd.DataFrame(incidents_encoded, columns=incidents)
X = pd.concat([X, X_incidents], axis=1)
del X['incident_type']
X = X.groupby(['sim_resident', 'neighborhood', 'month']).sum()
df_X = X.copy()
df_X.reset_index(['neighborhood', 'month'], inplace=True)

enc = LabelEncoder()
v_neighborhoods = np.unique(df_X['neighborhood'])
v_months = np.unique(df_X['month'])
feats = []
for n, m in itertools.product(v_neighborhoods, v_months):
    feats.append('{}_{}'.format(n, m))
feats = np.array(feats)
enc.fit(feats)

y = df_X[np.arange(len(incidents))].values
df_X['feat'] = df_X['neighborhood'].astype(str) + '_' + df_X['month'].astype(str)
X = df_X['feat'].values
X = enc.transform(X)[:, np.newaxis]

residents = df_X.index.values
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

batch_size = 514
num_epochs = 1000
criterion = L2Norm

dataset = SimpleDataset(X, y)
if sampler is None:
    dataloader_train = DataLoader(dataset, batch_size=batch_size, shuffle=True)
else:
    dataloader_train = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    num_epochs *= 5
dataloader_test = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# define model
num_layers = 2
model = Net_embedder(embedders, [embed_dim // 2] * num_layers + [1024], len(incidents))
model.to(device)

optimizer = torch.optim.RMSprop(model.parameters(), 1e-5)

best_loss = np.infty
best_model = None

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs * len(dataloader_train))

for i in range(num_epochs):
    # loss = train(model, dataloader_train, optimizer, criterion, device, scheduler=scheduler)
    loss = train(model, dataloader_train, optimizer, criterion, device, scheduler=None)
    scheduler.step(loss)

    print('{}: {:.4f}'.format(i, loss))

    if loss < best_loss:
        best_loss = loss
        best_model = copy.deepcopy(model)

        _, outputs = evaluate(best_model, dataloader_test, criterion, device)
        outputs = outputs.clone().detach().numpy()

        df_submission = pd.DataFrame(outputs, columns=incidents)
        df_submission['neighborhood'] = df_X['neighborhood'].values
        df_submission['month'] = df_X['month'].values
        df_submission = df_submission.groupby(['neighborhood', 'month']).sum().reset_index()

        # TODO: some garbage code to deal with missing value
        feats_submission = df_submission['neighborhood'].astype(str) + '_' + df_submission['month'].astype(str)
        feats_submission = feats_submission.values
        missing_rows = [x for x in feats if x not in feats_submission]
        for row in missing_rows:
            row = row.split('_')
            neighborhood, month = int(row[0]), int(row[1])
            new_row = 1e-8 * np.ones(df_submission.shape[1])
            new_row[0] = neighborhood
            new_row[1] = month
            df_submission.loc[len(df_submission)] = new_row
        df_submission.sort_values(['neighborhood', 'month'], inplace=True)
        df_submission.set_index(['neighborhood', 'month'], inplace=True)

        outputs = df_submission.values.copy()
        scores, penalties = get_score(df_ground_truth.values, outputs.astype(int))
        print('{:.3f}'.format(scores.sum()))
        print(penalties.mean(axis=0))

        outputs = df_submission.values.copy()
        outputs[outputs < 0] = 0
        scores, penalties = get_score(df_ground_truth.values, outputs.astype(int))
        print('{:.3f}'.format(scores.sum()))
        print(penalties.mean(axis=0))

        outputs = df_submission.values.copy()
        mins = outputs.min(axis=1)[:, np.newaxis]
        maxes = outputs.max(axis=1)[:, np.newaxis]
        outputs = outputs - mins
        outputs = outputs * maxes / (maxes - mins)
        outputs = np.nan_to_num(outputs)
        scores, penalties = get_score(df_ground_truth.values, outputs.astype(int))
        print('{:.3f}'.format(scores.sum()))
        print(penalties.mean(axis=0))



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
#                                noise_multiplier=4.0,
#                                max_grad_norm=1,
#                                target_delta=delta)
# privacy_engine.attach(optimizer)
#
# # create scheduler after privacy engine attaches
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs * len(dataloader_train))
#
# # initial result
# for i in range(num_epochs):
#     loss = train(model, dataloader_train, optimizer, criterion, device, scheduler=scheduler)
#     epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(delta)
#     print(
#         f"Train Epoch: {i} \t"
#         f"Loss: {loss:.6f} "
#         f"(ε = {epsilon:.2f}, δ = {delta}) for α = {best_alpha}")
#
#     if epsilon > max_epsilon:
#         break
#
#