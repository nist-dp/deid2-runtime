import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, Normalizer, OneHotEncoder

import sys  
sys.path.insert(0, '../runtime/scripts')
import metric, score

from utils_nn import *

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

# import data
df_incidents = pd.read_csv('../data/incidents.csv')

INDEX_COLS = ["epsilon", "neighborhood", "year", "month"]
df_submission_format = pd.read_csv('../data/submission_format.csv', index_col=INDEX_COLS)

df_ground_truth_all = score.get_ground_truth(df_incidents, df_submission_format)
df_ground_truth = df_ground_truth_all.loc[1.0]

df_ground_truth_normalized = df_ground_truth / df_ground_truth.sum(axis=1)[:, np.newaxis]
df_ground_truth_normalized.fillna(0, inplace=True)

# extra neighborhood data
df_coordinates = pd.read_csv('coordinates.csv')
for col in ['lat', 'lng']:
    df_coordinates[col] -= df_coordinates[col].mean()
df_temperatures = pd.read_csv('temperatures.csv')
for col in ['high', 'low']:
    df_temperatures[col] -= df_temperatures[col].mean()
for col in ['over_70', 'over_80', 'over_90', 'under_10', 'under_20', 'under_32']:
    df_temperatures[col] /= 30

# get X, y
y = df_ground_truth_normalized.values

X = df_ground_truth.index.to_frame().reset_index(drop=True)
X = pd.merge(X, df_coordinates, left_on=['neighborhood'], right_on=['code'])
X = pd.merge(X, df_temperatures, left_on=['month'], right_on=['month'])
for col in ['year', 'name', 'code']:
    del X[col]

cols_additional = ['lat', 'lng', 'high', 'low', 'over_70', 'over_80', 'over_90', 'under_10', 'under_20', 'under_32']
X_additional = X[cols_additional].values
norm = Normalizer()
X_additional = norm.fit_transform(X_additional)


X['feat'] = X['neighborhood'].astype(str) + '_' + X['month'].astype(str)
enc = LabelEncoder()
X = enc.fit_transform(X['feat'])[:, np.newaxis]
print(X.shape)

embed_dim = 1024
num_values = X.max()+1
embedders = []
embedder = nn.Embedding(num_values, embed_dim).to(device)
embedders.append(embedder)

batch_size = 128
num_epochs = 1000
criterion = LPNorm

dataset = SimpleDataset(X, y)
dataloader_train = DataLoader(dataset, batch_size=batch_size, shuffle=True)
dataloader_test = DataLoader(dataset, batch_size=1024, shuffle=False)

# define model
# model = Net(X.shape[1], [1024, 1024], y.shape[1])
model = Net_embedder(embedders, [1024, 1024], y.shape[1])
# model = Net_embedder(embedders, [1024, 1024], y.shape[1], additional_lookup=torch.tensor(X_additional).float().to(device))
# model.to(device)

optimizer = torch.optim.RMSprop(model.parameters(), 5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs * len(dataloader_train), eta_min=1e-8)

for i in range(num_epochs):
    loss = train(model, dataloader_train, optimizer, criterion, device, scheduler=scheduler, alpha=1 - i / num_epochs)
    # scheduler.step()
    print('{}: {:.4f}'.format(i, loss))

loss, logits = evaluate(model, dataloader_test, criterion, device)

probs = nn.functional.softmax(logits, dim=1)
probs = probs.clone().detach().numpy()
df_submission = get_submission(probs, df_ground_truth, df_submission_format)
scores, penalties = get_score(df_ground_truth_all.values, df_submission.values)
print('{:.3f}'.format(scores.sum()))
print(penalties.mean(axis=0))

probs = nn.functional.softmax(logits, dim=1)
probs = probs.clone().detach().numpy()
# probs[(probs < 0.05) & (probs > 0.04)]= 0
probs[probs < 0.05] = 0
df_submission = get_submission(probs, df_ground_truth, df_submission_format)
scores, penalties = get_score(df_ground_truth_all.values, df_submission.values)
print('{:.3f}'.format(scores.sum()))
print(penalties.mean(axis=0))

probs = nn.functional.softmax(logits, dim=1)
probs = probs.clone().detach().numpy()
probs[y < 0.05] = 0
df_submission = get_submission(probs, df_ground_truth, df_submission_format)
scores, penalties = get_score(df_ground_truth_all.values, df_submission.values)
print('{:.3f}'.format(scores.sum()))
print(penalties.mean(axis=0))

# X['month'] -= 1
# cols_feat = ['neighborhood', 'month']
# embedders = []
# for col in cols_feat:
#     num_values = len(X[col].unique())
#     embedder = nn.Embedding(num_values, num_values*3).to(device)
#     embedders.append(embedder)
# X = X[cols_feat].values

# enc = OneHotEncoder()
# cols_oh = ['neighborhood', 'month']
# X_oh = enc.fit_transform(X[cols_oh]).toarray()
# cols_oh = enc.get_feature_names(cols_oh)
# X_oh = pd.DataFrame(X_oh, columns=cols_oh)
#
# cols_noh = ['lat', 'lng']
# X_noh = X[cols_noh]
#
# # X = pd.concat([X_oh, X_noh], axis=1)
# X = X_oh
# X = X.values


#
#
# from opacus import PrivacyEngine
# privacy_engine = PrivacyEngine(model,
#                                batch_size=batch_size,
#                                sample_size=len(df_incidents) // 20,
#                                alphas=range(2, 32),
#                                noise_multiplier=1.1,
#                                max_grad_norm=1,)
# privacy_engine.attach(optimizer)
#
# # initial result
# for i in range(num_epochs):
#     loss = train(model, dataloader_train, optimizer, criterion, device, scheduler=scheduler, alpha=0)#1 - i / num_epochs)
#     scheduler.step()
#     print('{}: {:.4f}'.format(i, loss))
#
#     delta=1e-5
#     epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(delta)
#     print(
#         f"Train Epoch: {i} \t"
#         f"Loss: {np.mean(loss):.6f} "
#         f"(ε = {epsilon:.2f}, δ = {delta}) for α = {best_alpha}")