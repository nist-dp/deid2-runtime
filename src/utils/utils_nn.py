import itertools
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler

import pdb

class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class Net(nn.Module):
    def __init__(self, in_dim, hidden_dims, num_classes, dp_rate=0, bn=False):
        super(Net, self).__init__()
        self.fc_in = nn.Linear(in_dim, hidden_dims[0])
        self.fc_hidden = nn.ModuleList()
        for k in range(len(hidden_dims)-1):
            self.fc_hidden.append(nn.Linear(hidden_dims[k], hidden_dims[k+1]))
            self.fc_hidden.append(nn.Tanh())
            # self.fc_hidden.append(nn.ReLU())
            self.fc_hidden.append(nn.Dropout(dp_rate))
        self.fc_out = nn.Linear(hidden_dims[-1], num_classes)

    def forward(self, x):
        x = self.fc_in(x)
        for layers in self.fc_hidden:
            x = layers(x)
        x = self.fc_out(x)
        return x

class Net_embedder(nn.Module):
    def __init__(self, embedders, hidden_dims, num_classes, dp_rate=0, additional_lookup=None):
        super(Net_embedder, self).__init__()
        self.embedders = embedders
        self.additional_lookup = additional_lookup

        in_dim = np.array([e.embedding_dim for e in embedders]).sum()
        if additional_lookup is not None:
            in_dim += additional_lookup.shape[1]
        self.fc_in = nn.Linear(in_dim, hidden_dims[0])
        self.fc_hidden = nn.ModuleList()
        for k in range(len(hidden_dims)-1):
            self.fc_hidden.append(nn.Linear(hidden_dims[k], hidden_dims[k+1]))
            self.fc_hidden.append(nn.ReLU())
            self.fc_hidden.append(nn.Dropout(dp_rate))
        self.fc_out = nn.Linear(hidden_dims[-1], num_classes)

    def forward(self, x):
        out = []
        for i in range(len(self.embedders)):
            out.append(self.embedders[i](x[:, i]))
        out = torch.cat(out, axis=1)
        if self.additional_lookup is not None:
            additional = self.additional_lookup[x.flatten()]
            out = torch.cat([additional, out], axis=1)

        out = self.fc_in(out)
        for layers in self.fc_hidden:
            out = layers(out)
        out = self.fc_out(out)
        return out

def train(model, dataloader, optimizer, criterion, device, scheduler=None, alpha=None, reduction=None):
    model.train()
    losses = []
    for batch_idx, (X, y) in enumerate(tqdm(dataloader)):
        X, y = X.to(device), y.to(device)
        if y.type() == 'torch.cuda.DoubleTensor':
            y = y.float()
        # X, y = X.to(device).float(), y.to(device).float()

        logits = model(X)
        if alpha is not None:
            loss = criterion(logits, y, alpha=alpha)
        else:
            loss = criterion(logits, y)
        if reduction == 'mean':
            loss = loss.mean()
        if reduction == 'sum':
            loss = loss.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        losses.append(loss.item())

    loss = np.array(losses).mean()
    return loss

def evaluate(model, dataloader, criterion, device, reduction=None):
    model.eval()
    losses = []
    _logits = []
    for batch_idx, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        if y.type() == 'torch.cuda.DoubleTensor':
            y = y.float()
        # X, y = X.to(device).float(), y.to(device).float()

        logits = model(X)
        loss = criterion(logits, y)
        if reduction == 'mean':
            loss = loss.mean()
        if reduction == 'sum':
            loss = loss.sum()

        losses.append(loss.item())
        _logits.append(logits.clone().detach().cpu())
    loss = np.array(losses).mean()
    logits = torch.cat(_logits)
    return loss, logits

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
    return torch.norm(input - target, p=2, dim=1)
    # return torch.norm(input - target, p=2)

def L1Norm(input, target):
    return torch.norm(input - target, p=1, dim=1)

def LPNorm(logits, target, alpha=1):
    probs = F.softmax(logits, dim=1)
    loss = 0
    loss += alpha * torch.norm(probs - target, p=2) * 100
    loss += (1 - alpha) * torch.norm(probs - target, p=1)
    return loss

class UserSampler(Sampler):
    def __init__(self, user_to_idxs):
        self.idxs = list(user_to_idxs.values())
        self.lengths = np.array([len(x) for x in self.idxs])
        self.len_to_idxs = {}
        for length in np.unique(self.lengths):
            mask = self.lengths == length
            x = list(np.array(self.idxs)[mask])
            x = np.array(x)
            self.len_to_idxs[length] = x

    def get_sampled(self):
        sampled = []
        for length, idxs in self.len_to_idxs.items():
            if length == 1:
                sampled.append(idxs.flatten())
                continue

            mask = np.random.choice(length, size=len(idxs))
            idxs_sampled = idxs[np.arange(len(idxs)), mask]
            sampled.append(idxs_sampled)
        sampled = np.concatenate(sampled)
        return list(sampled)

    def __iter__(self):
        sampled = self.get_sampled()
        return iter(sampled)

    def __len__(self):
        return len(self.idxs)

def get_user_mapping(users):
    users_to_idxs = {}
    for i, user in enumerate(users):
        if user in users_to_idxs.keys():
            users_to_idxs[user].append(i)
        else:
            users_to_idxs[user] = [i]
    return users_to_idxs

def get_sampler(users):
    users_to_idxs = get_user_mapping(users)
    sampler = UserSampler(users_to_idxs)
    return sampler

class UserDataset(Dataset):
    def __init__(self, X, y, users_mapping):
        self.X = X
        self.y = y
        self.users_mapping = users_mapping

    def __len__(self):
        return len(self.users_mapping)

    def __getitem__(self, user_idx):
        idxs = self.users_mapping[user_idx]
        return self.X[idxs], self.y[idxs], np.ones(len(idxs)) / len(idxs)

def collate_variable_size(batch):
    result = []
    num_vars = len(batch[0])
    for i in range(num_vars):
        var = [x[i] for x in batch]
        var = list(itertools.chain(*var))
        result.append(torch.tensor(var))
    return result

def train_weighted(model, dataloader, optimizer, criterion, device, scheduler=None, alpha=None, reduction=None):
    model.train()
    losses = []
    for batch_idx, (X, y, weights) in enumerate(tqdm(dataloader)):
        X, y, weights = X.to(device), y.to(device), weights.to(device)
        if y.type() == 'torch.cuda.DoubleTensor':
            y = y.float()
        # X, y = X.to(device).float(), y.to(device).float()

        logits = model(X)
        if alpha is not None:
            loss = criterion(logits, y, alpha=alpha)
        else:
            loss = criterion(logits, y)
        loss = loss * weights
        if reduction == 'mean':
            loss = loss.mean()
        if reduction == 'sum':
            loss = loss.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        losses.append(loss.item())

    loss = np.array(losses).mean()
    return loss