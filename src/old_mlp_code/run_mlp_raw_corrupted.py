import os
import copy
import argparse
from opacus import PrivacyEngine

from utils.utils_general import *
from utils.utils_nn import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


def get_args():
    parser = argparse.ArgumentParser()

    # neural network arguments
    parser.add_argument('--embed_dim', type=int, default=1024)
    parser.add_argument('--hidden_dim', type=int, default=1024)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=514)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epsilon', type=float, default=1.0)
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
    df = df_ground_truth.reset_index()
    attr_feat = ['neighborhood', 'year', 'month']

    df['feat'] = df[attr_feat[0]].astype(str)
    for i in range(1, len(attr_feat)):
        df['feat'] = df['feat'] + '_' + df[attr_feat[i]].astype(str)
    enc, feats = get_encoder(df, attr_feat)

    y = df[np.arange(174).astype(str)].values
    y = y.sum(axis=1, keepdims=True)
    y = naively_add_laplace_noise(y, scale=20/args.epsilon).astype(float)

    X = enc.transform(df['feat'].values)[:, np.newaxis]

    # set up dataloaders
    dataset = SimpleDataset(X, y)
    dataloader_train = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # define model
    num_embeds = enc.classes_.shape[0]
    embed_dim = num_embeds * 1 if args.embed_dim is None else args.embed_dim
    embedders = [nn.Embedding(num_embeds, embed_dim).to(device)]

    hidden_dim = embed_dim * len(embedders) // 2 if args.hidden_dim is None else args.hidden_dim
    hidden_dims = [hidden_dim] * (args.num_layers - 1) + [1024]
    model = Net_embedder(embedders, hidden_dims, 1)
    print(model)
    model.to(device)

    num_epochs = 100
    criterion = nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), 1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs * len(dataloader_train))

    # Training loop
    best_loss = np.infty
    best_model = None
    missing_rows = None
    for i in range(num_epochs):
        loss = train(model, dataloader_train, optimizer, criterion, device, scheduler=scheduler)

        log = f"Train Epoch: {i}\tLoss: {loss:.6f}"
        print(log)

    ############################3

    # get data
    df_incidents, df_ground_truth, df_submission_format = get_data()
    df_incidents, incidents = get_incident_sums(df_incidents)
    df_ground_truth = df_ground_truth.loc[1.0]

    # get X, y matrices
    attr_feat = ['neighborhood', 'year', 'month']

    df_incidents['feat'] = df_incidents[attr_feat[0]].astype(str)
    for i in range(1, len(attr_feat)):
        df_incidents['feat'] = df_incidents['feat'] + '_' + df_incidents[attr_feat[i]].astype(str)

    y = df_incidents[np.arange(len(incidents))].values
    X = enc.transform(df_incidents['feat'].values)[:, np.newaxis]
    sampler = None if args.unif_sampling else get_sampler(df_incidents['sim_resident'].values)

    # set up dataloaders
    dataset = SimpleDataset(X, y)
    dataloader_train = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=args.unif_sampling)
    dataloader_test = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # define model
    model.fc_out = nn.Linear(1024, len(incidents)).to(device)
    model.to(device)

    criterion = L2Norm
    optimizer = torch.optim.RMSprop(model.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs * len(dataloader_train))

    # Training loop
    best_loss = np.infty
    best_model = None
    missing_rows = None
    for i in range(args.num_epochs):
        loss = train(model, dataloader_train, optimizer, criterion, device, scheduler=scheduler)

        log = f"Train Epoch: {i}\tLoss: {loss:.6f}"
        print(log)

        if loss < best_loss:
            best_loss = loss
            best_model = copy.deepcopy(model)

            _, outputs = evaluate(best_model, dataloader_test, criterion, device)
            outputs = outputs.clone().detach().numpy()

            df_submission = pd.DataFrame(outputs, columns=incidents)
            df_submission['neighborhood'] = df_incidents['neighborhood'].values
            df_submission['year'] = df_incidents['year'].values
            df_submission['month'] = df_incidents['month'].values
            df_submission = df_submission.groupby(['neighborhood', 'year', 'month']).sum().reset_index()

            # TODO: some garbage code to deal with missing value
            feats_submission = df_incidents['feat'].values
            missing_rows = list(set(feats) - set(feats_submission))
            for row in missing_rows:
                neighborhood, year, month = [int(x) for x in row.split('_')]
                new_row = 1e-8 * np.ones(df_submission.shape[1])
                new_row[:3] = [neighborhood, year, month]
                df_submission.loc[len(df_submission)] = new_row
            df_submission.sort_values(['neighborhood', 'year', 'month'], inplace=True)
            df_submission.set_index(['neighborhood', 'year', 'month'], inplace=True)

            outputs = df_submission.values.copy()
            outputs[outputs < 0] = 0
            scores, penalties = get_score(df_ground_truth.values, np.round(outputs).astype(int))
            print('{:.3f}'.format(scores.sum()))
            print(penalties.mean(axis=0))

            # outputs = df_submission.values.copy()
            # mins = outputs.min(axis=1)[:, np.newaxis]
            # maxes = outputs.max(axis=1)[:, np.newaxis]
            # outputs = outputs - mins
            # outputs = outputs * maxes / (maxes - mins)
            # outputs = np.nan_to_num(outputs)
            # scores, penalties = get_score(df_ground_truth.values, outputs.astype(int))
            # print('{:.3f}'.format(scores.sum()))
            # print(penalties.mean(axis=0))

