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
    # df_submission = df_submission.astype(int)

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

class LSTMClassifier(nn.Module):
    def __init__(self, embedder, hidden_size, num_classes, num_layers=2, bidirectional=True):
        super(LSTMClassifier, self).__init__()
        self.embedder = embedder
        intput_size = embedder.embedding_dim + num_classes
        self.lstm = nn.LSTM(input_size=intput_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional)
        self.fc_out = nn.Linear(hidden_size * num_layers, num_classes)

    def forward(self, input):
        X, y_prev = input
        seq_len = y_prev.shape[1]

        X = embedder(X).unsqueeze(1)
        X = X.repeat(1, seq_len, 1)
        X = torch.cat([X, y_prev], axis=-1)

        out, hidden = self.lstm(X)
        out = self.fc_out(out)
        return out

def train_lstm(model, dataloader, optimizer, criterion, device, scheduler=None):
    model.train()
    losses = []
    for batch_idx, (X, y) in enumerate(dataloader):
        X, y = X.to(device).long(), y.to(device).float()
        y_prev = torch.zeros(y[:, 0].shape).to(device)
        y_prev = y_prev.unsqueeze(1)
        y_prev = torch.cat([y_prev, y[:, :-1]], axis=1)

        logits = model((X, y_prev))
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        losses.append(loss.item())
    loss = np.array(losses).mean()
    return loss

def evaluate_lstm(model, dataloader, criterion, device):
    model.eval()
    losses = []
    _logits = []
    for batch_idx, (X, y) in enumerate(dataloader):
        X, y = X.to(device).float(), y.to(device).float()
        y_prev = torch.ones(y[:, 0].unsqueeze(1).shape).to(device)
        y_prev = torch.cat([y_prev, y[:, :-1]], axis=1)
        X = torch.cat([X, y_prev], axis=-1)

        logits = model(X)
        loss = criterion(logits, y)

        losses.append(loss.item())
        _logits.append(logits.clone().detach().cpu())
    loss = np.array(losses).mean()
    logits = torch.cat(_logits)
    return loss, logits


# import data
df_incidents = pd.read_csv('../data/incidents.csv')

INDEX_COLS = ["epsilon", "neighborhood", "year", "month"]
df_submission_format = pd.read_csv('../data/submission_format.csv', index_col=INDEX_COLS)

df_ground_truth_all = score.get_ground_truth(df_incidents, df_submission_format)
df_ground_truth = df_ground_truth_all.loc[1.0]

# no_incidents_mask = df_ground_truth.sum(axis=1) > 0
# zero_idxs = np.where(~no_incidents_mask.values)[0]
# df_ground_truth = df_ground_truth[no_incidents_mask]

df_ground_truth_normalized = df_ground_truth / df_ground_truth.sum(axis=1)[:, np.newaxis]
df_ground_truth_normalized.fillna(0, inplace=True)


df = df_ground_truth_normalized.reset_index('year', drop=True)
neighborhoods = df.index.levels[0].values
X, y = [], []
for n in neighborhoods:
    X.append(n)
    y.append(df.loc[n].values)
X = np.array(X)
y = np.array(y)

# enc = OneHotEncoder()
# X = enc.fit_transform(X.reshape(-1, 1)).toarray()
# X = np.tile(X, (12, 1, 1)).transpose(1, 0, 2)

dataset = SimpleDataset(X, y)
dataloader_train = DataLoader(dataset, batch_size=64, shuffle=True)
dataloader_test = DataLoader(dataset, batch_size=64, shuffle=False)

num_epochs = 500
# criterion = SoftCrossEntropyLoss
criterion = L2Norm

lr = 1e-4

embedder = nn.Embedding(X.shape[0], 512)
embedder.to(device)
model = LSTMClassifier(embedder, hidden_size=256, num_classes=y.shape[-1])
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs * len(dataloader_train), eta_min=1e-8)

for i in range(num_epochs):
    loss = train_lstm(model, dataloader_train, optimizer, criterion, device, scheduler)
    print('{}: {:.4f}'.format(i, loss))

loss, logits = evaluate_lstm(model, dataloader_test, criterion, device)
logits = logits.reshape(-1, logits.shape[-1])

probs = nn.functional.softmax(logits, dim=1)
probs = probs.clone().detach().numpy()
df_submission = get_submission(probs, df_ground_truth, df_submission_format)
scores, penalties = get_score(df_ground_truth_all.values, df_submission.values)
print('{:.3f}'.format(scores.sum()))
print(penalties.mean(axis=0))



pdb.set_trace()








probs = nn.functional.softmax(logits, dim=1)
probs = probs.clone().detach().numpy()
probs[probs < 0.05] = 0
df_submission = get_submission(probs, df_ground_truth, df_submission_format, zero_idxs)
scores, penalties = get_score(df_ground_truth_all.values, df_submission.values)
print('{:.3f}'.format(scores.sum()))
print(penalties.mean(axis=0))

probs = nn.functional.softmax(logits, dim=1)
probs = probs.clone().detach().numpy()
probs[y < 0.05] = 0
df_submission = get_submission(probs, df_ground_truth, df_submission_format, zero_idxs)
scores, penalties = get_score(df_ground_truth_all.values, df_submission.values)
print('{:.3f}'.format(scores.sum()))
print(penalties.mean(axis=0))

pdb.set_trace()