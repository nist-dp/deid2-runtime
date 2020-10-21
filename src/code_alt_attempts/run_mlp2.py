import os
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
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

def L2Norm(logits, target):
    probs = F.softmax(logits, dim=1)
    return torch.norm(probs - target, p=2)

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

X = df_ground_truth.reset_index().melt(['neighborhood', 'year', 'month'])
X = X.rename(columns={'variable': 'incident'})
y = X['value'].values.astype(float)[:, np.newaxis]

# X['feat'] = ''
# for col in ['neighborhood', 'month', 'incident']:
#     X['feat'] = X['feat'] + '_' + X[col].values.astype(str)
# from sklearn.preprocessing import LabelEncoder
# enc = LabelEncoder()
# X = enc.fit_transform(X['feat'])[:, np.newaxis]
#
# embed_dim = 2048
# num_values = X.max()+1
# embedders = []
# embedder = nn.Embedding(num_values, embed_dim).to(device)
# embedders.append(embedder)

X['month'] -= 1
cols_feat = ['neighborhood', 'month', 'incident']
embedders = []
for col in cols_feat:
    num_values = len(X[col].unique())
    embedder = nn.Embedding(num_values, num_values * 4).to(device)
    embedders.append(embedder)
X = X[cols_feat].values.astype(int)

batch_size = 4096
num_epochs = 500
criterion = torch.nn.MSELoss()

dataset = SimpleDataset(X, y)
dataloader_train = DataLoader(dataset, batch_size=batch_size, shuffle=True)
dataloader_test = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# define model
model = Net_embedder(embedders, [1024, 1024], 1)
model.to(device)

optimizer = torch.optim.RMSprop(model.parameters(), 1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs * len(dataloader_train), eta_min=1e-8)

best_loss = np.infty
best_model = None
for i in tqdm(range(num_epochs)):
    loss = train(model, dataloader_train, optimizer, criterion, device, scheduler=scheduler)
    print('{}: {:.4f}'.format(i, loss))

    if loss < best_loss:
        best_model = copy.deepcopy(model)

    if i+1 == num_epochs or (i+1) % 10 == 0:
        loss, outputs = evaluate(best_model, dataloader_test, criterion, device)
        outputs = outputs.numpy()

        df_submission = pd.DataFrame(X.copy(), columns=cols_feat)
        df_submission['pred'] = np.maximum(0, outputs).astype(int)
        df_submission['month'] += 1
        df_submission = df_submission.pivot_table(index=['neighborhood', 'month'], columns=['incident'], values='pred')
        df_submission.columns.name = None
        df_submission_eps = df_submission.copy()

        df_submission = df_submission_format.copy()
        epsilons = df_submission.index.levels[0]
        for eps in epsilons:
            df_submission.loc[eps, :] = df_submission_eps.values

        scores, penalties = get_score(df_ground_truth_all.values, df_submission.values)
        print('{:.3f}'.format(scores.sum()))
        print(penalties.mean(axis=0))

pdb.set_trace()

# dp1 = df_submission.copy().values

probs = nn.functional.softmax(logits, dim=1)
probs = probs.clone().detach().numpy()
# probs[(probs < 0.05) & (probs > 0.03)] = 0
probs[probs < 0.05] = 0
df_submission = get_submission(probs, df_ground_truth, df_submission_format)
scores, penalties = get_score(df_ground_truth_all.values, df_submission.values)
print('{:.3f}'.format(scores.sum()))
print(penalties.mean(axis=0))

# dp2 = df_submission.copy().values

probs = nn.functional.softmax(logits, dim=1)
probs = probs.clone().detach().numpy()
probs[y < 0.05] = 0
df_submission = get_submission(probs, df_ground_truth, df_submission_format)
scores, penalties = get_score(df_ground_truth_all.values, df_submission.values)
print('{:.3f}'.format(scores.sum()))
print(penalties.mean(axis=0))

# gt = df_ground_truth_all.copy().values

# for a in [dp1, dp2, gt]:
#     a_normalized = a.copy().astype(float)
#     for i in range(len(a)):
#         if (a[i,:] > 0).any():
#             a_normalized[i,:] = a[i,:] / a[i,:].sum()
#     print(a_normalized.sum(axis=1))

# ((gt == 0) & (dp1 > 0)).sum() * 0.2
# ((gt == 0) & (dp2 > 0)).sum() * 0.2
# x = dp1[dp1 != dp2]
# np.where(x >= 0.05, x, 0)

# pdb.set_trace()

# for i in range(len(gt)):
#     print(scorer._penalty_components(gt[i, :], dp1[i, :]))
#     print(scorer._penalty_components(gt[i, :], dp2[i, :]))
#     print()

pdb.set_trace()


# from opacus import PrivacyEngine
# privacy_engine = PrivacyEngine(model, 
#                                batch_size=batch_size, 
#                                sample_size=len(df_incidents)//20,  
#                                alphas=range(2, 32), 
#                                noise_multiplier=1.0, 
#                                max_grad_norm=1,)
# privacy_engine.attach(optimizer)

# # initial result
# for i in range(num_epochs):
#     loss = train(model, dataloader_train, optimizer, criterion, device, scheduler=scheduler)
#     scheduler.step()
#     print('{}: {:.4f}'.format(i, loss))

#     delta=1e-5
#     epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(delta) 
#     print(
#         f"Train Epoch: {i} \t"
#         f"Loss: {np.mean(loss):.6f} "
#         f"(ε = {epsilon:.2f}, δ = {delta}) for α = {best_alpha}")