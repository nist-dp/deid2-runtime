import os
import json
import copy
import argparse
import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from opacus import PrivacyEngine

import sys  
sys.path.insert(0, '../runtime/scripts')
import metric, score

from utils.utils_nn import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPS_RUN = {1.0: 0,
           2.0: 1,
           10.0: 12
           }
scorer = metric.Deid2Metric()

def get_data():
    df_incidents = pd.read_csv('../data/incidents.csv')
    df_submission_format = pd.read_csv('../data/submission_format.csv',
                                       index_col=["epsilon", "neighborhood", "year", "month"])
    df_ground_truth = score.get_ground_truth(df_incidents, df_submission_format)
    return df_incidents, df_ground_truth, df_submission_format

def get_encoder(attr_feat):
    unique_values = []
    for attr in attr_feat:
        unique_values.append(list(np.unique(df_incidents[attr])))

    feats = []
    for feat in itertools.product(*unique_values):
        feats.append('_'.join([str(f) for f in feat]))
    feats = np.array(feats)

    enc = LabelEncoder()
    enc.fit(feats)

    return enc, feats

def get_sampler(residents):
    resident_to_idxs = {}
    for i, resident in enumerate(residents):
        if resident in resident_to_idxs.keys():
            resident_to_idxs[resident].append(i)
        else:
            resident_to_idxs[resident] = [i]
    sampler = UserSampler(resident_to_idxs)
    return sampler

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

    return scores, raw_penalties

def get_submission(probs, df_ground_truth, df_submission_format):
    sums = df_ground_truth.sum(axis=1).values
    hist = (sums[:, np.newaxis] * probs).astype(int)

    df_submission = df_submission_format.copy()
    epsilons = df_submission.index.levels[0]
    for eps in epsilons:
        df_submission.loc[eps, :] = hist

    return df_submission

def get_args():
    parser = argparse.ArgumentParser()

    # neural network arguments
    parser.add_argument('--embed_dim', type=int, default=None)
    parser.add_argument('--hidden_dim', type=int, default=None)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=139)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    # privacy arguments
    parser.add_argument('--epsilon', type=float, default=None)
    parser.add_argument('--noise_multiplier', type=float, default=1.5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    # misc arguments
    parser.add_argument('--unif_sampling', action='store_true')

    args = parser.parse_args()
    print(args)
    return args

if __name__ == '__main__':
    args = get_args()

    # get data
    df_incidents, df_ground_truth, df_submission_format = get_data()
    df_ground_truth = df_ground_truth.loc[1.0]

    # get X, y matrices
    attr_feat = ['neighborhood', 'year', 'month']

    df_incidents['feat'] = df_incidents[attr_feat[0]].astype(str)
    for i in range(1, len(attr_feat)):
        df_incidents['feat'] = df_incidents['feat'] + '_' + df_incidents[attr_feat[i]].astype(str)
    enc, feats = get_encoder(attr_feat)

    y = df_incidents['incident_type'].values
    X = enc.transform(df_incidents['feat'].values)[:, np.newaxis]
    sampler = None if args.unif_sampling else get_sampler(df_incidents['sim_resident'].values)

    # set up dataloaders
    num_embeds = enc.classes_.shape[0]
    embed_dim = num_embeds * 1 if args.embed_dim is None else args.embed_dim
    embedders = [nn.Embedding(num_embeds, embed_dim).to(device)]

    dataset = SimpleDataset(X, y)
    dataloader_train = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=args.unif_sampling)

    X_test = enc.transform(feats)[:, np.newaxis]
    dataset_test = SimpleDataset(X_test, y[:len(X_test)]) # y doesn't matter
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)

    # define model
    num_classes = y.max() - y.min() + 1
    hidden_dim = embed_dim // 2 if args.hidden_dim is None else args.hidden_dim
    hidden_dims = [embed_dim // 2] * (args.num_layers - 1) + [1024]
    model = Net_embedder(embedders, hidden_dims, num_classes)
    print(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), args.lr)
    if args.epsilon is not None:
        max_epsilon, delta, sensitivity = get_priv_params(args.epsilon)
        privacy_engine = PrivacyEngine(model,
                                       batch_size=args.batch_size,
                                       sample_size=len(sampler),
                                       alphas=list(range(2, 32)),
                                       noise_multiplier=args.noise_multiplier,
                                       max_grad_norm=args.max_grad_norm,
                                       target_delta=delta)
        privacy_engine.attach(optimizer)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs * len(dataloader_train))
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    # Training loop
    best_loss = np.infty
    best_model = None
    for i in range(args.num_epochs):
        loss = train(model, dataloader_train, optimizer, criterion, device, scheduler=scheduler)

        log = f"Train Epoch: {i}\tLoss: {loss:.6f}"
        if args.epsilon is not None:
            epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(delta)
            log += f" (ε = {epsilon:.2f}, δ = {delta}) for α = {best_alpha}"
            if epsilon > max_epsilon:
                break
        print(log)

        if loss < best_loss:
            best_loss = loss
            best_model = copy.deepcopy(model)

            _, outputs = evaluate(model, dataloader_test, criterion, device)
            probs = nn.functional.softmax(outputs, dim=1)
            probs = probs.clone().detach().numpy()
            sums = df_ground_truth.sum(axis=1).values[:, np.newaxis]
            outputs = (probs * sums).astype(int)
            outputs[outputs < 0] = 0

            scores, penalties = get_score(df_ground_truth.values, outputs)
            print('{:.3f}'.format(scores.sum()))
            print(penalties.mean(axis=0))

            # outputs[probs < scorer.threshold] = 0
            # scores, penalties = get_score(df_ground_truth.values, outputs)
            # print('{:.3f}'.format(scores.sum()))
            # print(penalties.mean(axis=0))








# feats_idx = [x.split('_') for x in feats]
# feats_idx = [[int(x[0]), int(x[1])] for x in feats_idx]
# feats_idx = np.array(feats_idx)
#
# df_submission = pd.DataFrame()
# df_submission['neighborhood'] = feats_idx[:, 0]
# df_submission['mood'] = feats_idx[:, 1]
# for i in range(num_classes):
#     df_submission[i] = np.zeros(len(df_submission))
# df_submission.set_index(['neighborhood', 'mood'], inplace=True)
