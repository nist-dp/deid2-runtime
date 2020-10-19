import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

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

def L2Norm(outputs, target):
    return torch.norm(outputs - target, p=2)

def get_submission(outputs, df_submission_format):
    df_submission = df_submission_format.copy()
    epsilons = df_submission.index.levels[0]
    for eps in epsilons:
        df_submission.loc[eps, :] = outputs

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

# get X, y
scale = 1
y = df_ground_truth.values / scale

X = df_ground_truth.index.to_frame().reset_index(drop=True)
del X['year']

X['feat'] = ''
for col in ['neighborhood', 'month']:
    X['feat'] = X['feat'] + '_' + X[col].values.astype(str)
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
X = enc.fit_transform(X['feat'])[:, np.newaxis]

embed_dim = 1024
num_values = X.max()+1
embedders = []
embedder = nn.Embedding(num_values, embed_dim).to(device)
embedders.append(embedder)

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

print(X.shape)

dataset = SimpleDataset(X, y)
dataloader_train = DataLoader(dataset, batch_size=100, shuffle=True)
dataloader_test = DataLoader(dataset, batch_size=100, shuffle=False)

num_epochs = 100
criterion = L2Norm

# define model
model = Net_embedder(embedders, [1024, 1024], y.shape[1])
# model = Net(X.shape[1], [1024, 1024], y.shape[1])
model.to(device)

optimizer = torch.optim.RMSprop(model.parameters(), 5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs * len(dataloader_train), eta_min=1e-8)

# initial result
for i in range(num_epochs):
    loss = train(model, dataloader_train, optimizer, criterion, device)
    scheduler.step()
    print('{}: {:.4f}'.format(i, loss))

loss, outputs = evaluate(model, dataloader_test, criterion, device)
outputs *= scale
outputs = outputs.clone().detach().numpy()
outputs = outputs.astype(int)
outputs = np.where(outputs > 0, outputs, 0)
df_submission = get_submission(outputs, df_submission_format)
scores, penalties = get_score(df_ground_truth_all.values, df_submission.values)
print('{:.3f}'.format(scores.sum()))
print(penalties.mean(axis=0))

probs = y / y.sum(axis=1)[:, np.newaxis]
outputs[probs < 0.05] = 0
df_submission = get_submission(outputs, df_submission_format)
scores, penalties = get_score(df_ground_truth_all.values, df_submission.values)
print('{:.3f}'.format(scores.sum()))
print(penalties.mean(axis=0))