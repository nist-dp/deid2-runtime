import os
import copy
import argparse
from opacus import PrivacyEngine

from utils.utils_general import *
from utils.utils_nn import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_args():
    parser = argparse.ArgumentParser()

    # neural network arguments
    parser.add_argument('--embed_dim', type=int, default=1024)
    parser.add_argument('--hidden_dim', type=int, default=1024)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=514)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-5)
    # privacy arguments
    parser.add_argument('--epsilon', type=float, default=None, choices=[1.0, 2.0, 10.0])
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
    df_incidents, incidents = get_incident_sums(df_incidents)
    df_ground_truth = df_ground_truth.loc[1.0]

    # get X, y matrices
    attr_feat = ['neighborhood', 'year', 'month']

    df_incidents['feat'] = df_incidents[attr_feat[0]].astype(str)
    for i in range(1, len(attr_feat)):
        df_incidents['feat'] = df_incidents['feat'] + '_' + df_incidents[attr_feat[i]].astype(str)
    enc, feats = get_encoder(df_incidents, attr_feat)

    y = df_incidents[np.arange(len(incidents))].values
    X = enc.transform(df_incidents['feat'].values)[:, np.newaxis]
    users = df_incidents['sim_resident'].values
    users_to_idxs = get_user_mapping(users)

    # set up dataloaders
    dataset_train = UserDataset(X, y, users_to_idxs)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_variable_size)
    dataset_test = SimpleDataset(X, y)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)

    # define model
    num_embeds = enc.classes_.shape[0]
    embed_dim = num_embeds * 1 if args.embed_dim is None else args.embed_dim
    embedders = [nn.Embedding(num_embeds, embed_dim).to(device)]

    num_classes = y.max() - y.min() + 1
    hidden_dim = embed_dim * len(embedders) // 2 if args.hidden_dim is None else args.hidden_dim
    hidden_dims = [hidden_dim] * (args.num_layers - 1) + [1024]
    model = Net_embedder(embedders, hidden_dims, len(incidents))
    print(model)
    model.to(device)

    criterion = L2Norm
    optimizer = torch.optim.RMSprop(model.parameters(), args.lr)
    if args.epsilon is not None:
        max_epsilon, delta, sensitivity = get_priv_params(args.epsilon)
        privacy_engine = PrivacyEngine(model,
                                       batch_size=args.batch_size,
                                       sample_size=len(dataloader_train),
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
    missing_rows = None
    for i in range(args.num_epochs):
        loss = train_weighted(model, dataloader_train, optimizer, criterion, device, scheduler=scheduler, reduction='sum')
        # loss = train(model, dataloader_train, optimizer, criterion, device, scheduler=None)
        # scheduler.step(loss)

        log = f"Train Epoch: {i}\tLoss: {loss:.6f}"
        if args.epsilon is not None:
            epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(delta)
            log += f" (ε = {epsilon:.2f}, δ = {delta}) for α = {best_alpha}"
            # if epsilon > max_epsilon:
            #     break
        print(log)

        if loss < best_loss:
            best_loss = loss
            best_model = copy.deepcopy(model)

            _, outputs = evaluate(best_model, dataloader_test, criterion, device, reduction='mean')
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

            outputs = df_submission.values.copy()
            outputs[outputs < 0] = 0
            probs = outputs / outputs.sum(axis=1)[:, np.newaxis]
            outputs[probs < scorer.threshold] = 0
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



