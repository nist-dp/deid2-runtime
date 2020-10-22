from utils.utils_general import *
import pdb

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

# get data
df_incidents, df_ground_truth_all, df_submission_format = get_data()
df_ground_truth = df_ground_truth_all.loc[1.0]

SENSITIVTY = 20
MAX_EPSILON = 10
STEP = 0.05
df = []
for epsilon in np.arange(STEP, MAX_EPSILON+STEP, STEP):
    df_private = df_ground_truth.sum(axis=1).to_frame()
    df_private.columns = ['count']
    df_private = naively_add_laplace_noise(df_private, scale=SENSITIVTY / epsilon)
    df_private['epsilon'] = round(epsilon, 2)
    df.append(df_private.reset_index())
df = pd.concat(df)
cols = list(df.columns[-1:]) + list(df.columns[:-1])
df = df[cols]
df.sort_values(INDEX_COLS, inplace=True)
df.to_csv('./data/laplace_sums.csv')

# # just to check this matches the benchmark score
# for epsilon in [1.0, 2.0, 10.0]:
#     mask = df['epsilon'] == epsilon
#     outputs = df[mask]
#     outputs = outputs.set_index(INDEX_COLS).values
#     scores, penalties = get_score(df_ground_truth.values, outputs)
#     print('{:.3f}'.format(scores.sum()))
#     print(penalties.mean(axis=0))


