import argparse
from utils.utils_general import *
import pdb

with open('../data/parameters.json') as f:
    parameters = json.load(f)
runs = parameters["runs"]

df_incidents, df_ground_truth_all, df_submission_format = get_data()
df_incidents.reset_index(drop=True)
df_ground_truth = df_ground_truth_all.loc[1.0]

df_counts = df_incidents.groupby(['sim_resident']).size().to_frame()
df_counts.columns = ['count']

results = []
for sensitivity in np.arange(20) + 1:
    # mask = df_counts['count'] <= s
    # residents = df_counts[mask].index.values
    # mask = df_incidents['sim_resident'].isin(residents)
    # df_private = df_incidents[mask].reset_index(drop=True)

    df_private = df_incidents.sample(frac=1).reset_index() # randomly shuffle rows
    df_private = df_private.groupby(['sim_resident']).head(sensitivity) # take first <sensitivity> rows from each resident

    frac_kept = df_private.shape[0] / df_incidents.shape[0]

    df_private = score.get_ground_truth(df_private, df_submission_format)
    df_private = df_private.loc[1.0]

    print("Sensitivity: {}, frac: {:.4f}".format(sensitivity, frac_kept))
    laplace_score, gaussian_score = 0, 0
    for r in runs:
        epsilon = r["epsilon"]
        delta = r["delta"]

        # Laplace Mechanism
        scale = sensitivity / epsilon
        df_outputs = df_private.copy()
        df_outputs = naively_add_laplace_noise(df_outputs, scale=scale)
        outputs = df_outputs.values
        scores, penalties = get_score(df_ground_truth.values, outputs)
        laplace_score += scores.sum()

        # Gaussian Mechanism
        scale = np.sqrt(sensitivity * 2 * np.log(2 / delta)) / epsilon
        df_outputs = df_private.copy()
        df_outputs = naively_add_gaussian_noise(df_outputs, scale=scale)
        outputs = df_outputs.values
        scores, penalties = get_score(df_ground_truth.values, outputs)
        gaussian_score += scores.sum()

    results.append([sensitivity,
                    round(frac_kept, 8),
                    round(laplace_score, 3),
                    round(gaussian_score, 3),
                    ])

    print('Laplace: {:.3f}'.format(laplace_score))
    print('Gaussian: {:.3f}'.format(gaussian_score))
    print('\n')

result_cols = ['sensitivity', 'frac_kept', 'laplace', 'gaussian']
df_results = pd.DataFrame(results, columns=result_cols)

pdb.set_trace()