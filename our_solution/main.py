import json
import warnings
from pathlib import Path
from typing import Optional

from tqdm import tqdm
import typer

import numpy as np
import pandas as pd
from loguru import logger

def get_df_user_type(df, cols_attr=['year', 'month', 'neighborhood', 'incident_type']):
    """
    Reformat the dataset so that each row corresponds to the number of users that fall under a user "type",
    where we define a user type to be the combination - (YEAR, MONTH, NEIGHBORHOOD, INCIDENT_TYPE, NUM_CALLS).
    In other worse, we can define a resident by a binary variable that represents whether or not
    they made <NUM_CALLS> calls of incident type <INCIDENT_TYPE> from neighborhood <NEIGHBORHOOD>
    during the time period <YEAR-MONTH>.

    Note that make a query: "How many residents fall under type = (YEAR, MONTH, NEIGHBORHOOD, INCIDENT_TYPE, NUM_CALLS)?",
    you can simply run the commmand:
    df_user_type.loc[(YEAR, MONTH, NEIGHBORHOOD, INCIDENT_TYPE, NUM_CALLS), 'num_calls']

    Therefore this dataframe also serves as a data structure that maps the above queries to the answers for the input df
    """
    df_user_type = df.groupby(['sim_resident'] + cols_attr).size().to_frame()
    df_user_type.columns = ['num_calls']
    df_user_type.reset_index(cols_attr, inplace=True)
    df_user_type = df_user_type.reset_index().groupby(cols_attr + ['num_calls']).size().to_frame()
    df_user_type.columns = ['count']
    return df_user_type

def naively_add_laplace_noise(arr, scale, seed=None, clip_and_round=True):
    """
    Add Laplace random noise of the desired scale to the dataframe of counts. Noisy counts will be
    clipped to [0,∞) and rounded to the nearest positive integer.
    Expects a numpy array and a Laplace scale.
    """
    if seed is not None:
        np.random.seed(seed)
    if isinstance(scale, np.ndarray):
        assert(scale.shape == arr.shape)
        noise = np.random.laplace(scale=scale)
    else:
        noise = np.random.laplace(scale=scale, size=arr.size).reshape(arr.shape)
    result = arr + noise
    if clip_and_round:
        result = np.clip(result, a_min=0, a_max=np.inf)
        result = result.round().astype(np.int)
    return result

# current numpy and pandas versions have a FutureWarning for a mask operation we use;
# we will ignore it
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT_DIRECTORY = Path("/codeexecution")
# ROOT_DIRECTORY = Path("../") # local testing
RUNTIME_DIRECTORY = ROOT_DIRECTORY / "submission"
DATA_DIRECTORY = ROOT_DIRECTORY / "data"

DEFAULT_SUBMISSION_FORMAT = DATA_DIRECTORY / "submission_format.csv"
DEFAULT_INCIDENTS = DATA_DIRECTORY / "incidents.csv"
DEFAULT_PARAMS = DATA_DIRECTORY / "parameters.json"
DEFAULT_OUTPUT = ROOT_DIRECTORY / "submission.csv"

# we add our public dataset in ./data_public (where our public dataset is just a copy of the development dataset for 2019)
DEFAULT_INCIDENTS_PUBLIC = "./data_public/incidents.csv"

def main(
    submission_format: Path = DEFAULT_SUBMISSION_FORMAT,
    incident_csv: Path = DEFAULT_INCIDENTS,
    output_file: Optional[Path] = DEFAULT_OUTPUT,
    params_file: Path = DEFAULT_PARAMS,
    incident_public_csv: Path = DEFAULT_INCIDENTS_PUBLIC,
):
    """
    Generate an example submission. The defaults are set so that the script will run successfully
    without being passed any arguments, invoked only as `python main.py`.
    """
    logger.info("loading parameters")
    params = json.loads(params_file.read_text())
    # calculate the Laplace scales for each run
    scales = {
        run["epsilon"]: run["max_records_per_individual"] / run["epsilon"]
        for run in params["runs"]
    }
    logger.info(f"laplace scales for each epsilon: {scales}")

    # read in the submission format
    logger.info(f"reading submission format from {submission_format} ...")
    df_submission_format = pd.read_csv(
        submission_format, index_col=["epsilon", "neighborhood", "year", "month"]
    )
    logger.info(f"read dataframe with {len(df_submission_format):,} rows")

    # read in the raw incident data
    logger.info(f"reading raw incident data from {incident_csv} ...")
    df_incidents = pd.read_csv(incident_csv, index_col=0)
    logger.info(f"read dataframe with {len(df_incidents):,} rows")

    # read in the raw public incident data
    logger.info(f"reading raw public incident data from {incident_public_csv} ...")
    df_incidents_public = pd.read_csv(incident_public_csv, index_col=0)
    logger.info(f"read dataframe with {len(df_incidents_public):,} rows")

    """
    Set up data and queries
    """
    # set public and private datasets
    df_private = df_incidents
    df_public = df_incidents_public

    # convert to dataframe of "user types"
    cols_attr = ['year', 'month', 'neighborhood', 'incident_type']
    df_user_type_private = get_df_user_type(df_private, cols_attr)
    df_user_type_public = get_df_user_type(df_public, cols_attr)

    # use public dataset to get queries
    df_queries = df_user_type_public.copy()

    # change the year of each query to match the schema (df_submission_format) and therefore the year of private dataset.
    year_submission = df_submission_format.index.levels[2][0]
    df_queries.reset_index(inplace=True)
    df_queries['year'] = year_submission
    df_queries.set_index(cols_attr + ['num_calls'], inplace=True)
    df_queries.sort_index(inplace=True)
    assert(df_queries.index.levels[0][0] == df_user_type_private.index.levels[0][0])

    """
    We expand our base set of queries in the following way to give us more coverage:
    
    1) add adjacent months
    For each query in df_queries, we duplicate the query and change the month field. Our motivation is that if people make 
    some type of call during a certain month, we should also check if people make the same call in the preceding/following month(s).
    For example, suppose query A has month=June. We then duplicate the query, keeping the fields the same except we change 
    month to July. 
    
    2) add extra 'num calls'
    For each query in df_queries, we duplicate the query and change the num_calls field. Our motivation is that if there
    exists a person who make a certain type of call X times, we should also check if there are other people who make the 
    same type of call Y times (where Y is some small number that we glean from the public dataset).
    For example, suppose query A has NUM_CALLS=5. We then duplicate query, keeping the fields the same except we change 
    NUM_CALLS to 1. 
    """
    # add adjacent months to queries
    df_queries.reset_index(inplace=True)
    df_extra = [df_queries]
    delta = [1, 2, 3] # we add the preceding and following 3 months for each query
    delta += [12 - d for d in delta]
    for x in delta:
        df = df_queries.copy()
        df['month'] -= 1  # scales months to values from 0-11
        df['month'] = (df['month'] + x) % 12
        df['month'] += 1  # changes it back to values 1-12
        df_extra.append(df)

        # testing
        test = np.unique((df['month'] - df_queries['month'] + 12) % 12)
        assert (len(test) == 1)
        assert (test[0] == x)

    df_queries = pd.concat(df_extra)
    df_queries.set_index(cols_attr + ['num_calls'], inplace=True)

    # add extra num_calls
    df_queries.reset_index(inplace=True)
    mask = (df_queries['count'] > 0) & (df_queries['num_calls'] > 1)
    df = df_queries[mask]

    df_extra = [df_queries]
    for num_calls in np.arange(2) + 1: # we duplicate all queries and set NUM_CALLS={1, 2}
        df = df_queries.copy()
        df.loc[:, 'num_calls'] = num_calls
        df_extra.append(df)
    df_queries = pd.concat(df_extra)
    df_queries.set_index(cols_attr + ['num_calls'], inplace=True)

    # remove duplicate queries that were created from the above steps
    df_queries['count'] = 0
    df_queries.reset_index(inplace=True)
    df_queries.drop_duplicates(inplace=True)
    df_queries.set_index(cols_attr + ['num_calls'], inplace=True)
    df_queries.sort_index(inplace=True)

    """
    We now query the private dataset using a LEFT JOIN.
    Note that each row of df_queries corresponds to a query, where  the index of df_queries are the attributes for each query.
    By joining with the private dataset (df_user_type_private), we get the answer for each query.
    """
    df_queries = pd.merge(df_queries, df_user_type_private, how='left', left_index=True, right_index=True, suffixes=['_public', ''])
    df_queries = df_queries[['count']]
    df_queries.fillna(0, inplace=True) # if nan, the user type corresponding the query doesn't exist in the private dataset

    """
    We apply the Laplace mechanism to the answer of each query, which is found in df_queries['count'].
    Since each query is asking "how many users are of type=(YEAR, MONTH, NEIGHBORHOOD, INCIDENT_TYPE, NUM_CALLS),
    each user contributes at most 1 to the answer of all queries, meaning our l1-sensitivity=1
    """
    SENSITIVITY = 1
    df_submission = df_submission_format.copy()
    incidents = df_submission_format.columns.values
    epsilons = df_submission_format.index.levels[0]
    for epsilon in epsilons: # we run the Laplace for different levels of epsilon specified in df_submission_format (generated from the params file)
        df_output = df_queries.copy()
        df_output = naively_add_laplace_noise(df_output, SENSITIVITY / epsilon)

        """
        df_output contains the number of residents who make calls with attributes=(YEAR, MONTH, NEIGHBORHOOD, INCIDENT_TYPE)
        <NUM_CALLS> times. To get the final count for each combination (YEAR, MONTH, NEIGHBORHOOD, INCIDENT_TYPE), 
        we take the product of <COUNT_USERS> (df_output['count'])  and <NUM_CALLS> (df_output['num_calls']).
        """
        df_output = df_output.reset_index()
        df_output['total_count'] = df_output['num_calls'] * df_output['count']
        df_output = pd.pivot_table(df_output, aggfunc='sum',
                                    values='total_count', columns=['incident_type'], index=['neighborhood', 'year', 'month'])
        df_output.fillna(0, inplace=True)
        df_output.columns = df_output.columns.values

        # add missing cols - if for some incident_type, total=0 for all (YEAR, MONTH, NEIGHBORHOOD), then it will be missing from the pivot table
        num_incidents = len(incidents)
        missing_cols = list(set(np.arange(num_incidents)) - set(df_output.columns.values))
        for col in missing_cols:
            df_output.loc[:, col] = 0
        df_output = df_output[np.arange(num_incidents)]

        # add missing rows - if for some (YEAR, MONTH, NEIGHBORHOOD), total=0 for all incident types, then it will be missing from the pivot table
        missing_rows_idxs = pd.merge(df_submission_format.loc[1.0], df_output, how='left', left_index=True, right_index=True)
        mask = missing_rows_idxs.isnull().any(axis=1)
        missing_rows_idxs = missing_rows_idxs[mask].reset_index()[['neighborhood', 'year', 'month']].values

        df_output.reset_index(inplace=True)
        row_template = np.zeros(shape=df_output.values[0].shape)
        for idx in missing_rows_idxs:
            row = row_template.copy()
            row[:len(idx)] = idx
            df_output.loc[len(df_output), :] = row

        # reformat df_output to match df_submission_format and add it to our final output - df_submission
        df_output = df_output.astype(int)
        df_output.sort_values(['neighborhood', 'year', 'month'], inplace=True)
        df_output['epsilon'] = epsilon
        df_output.set_index(['epsilon', 'neighborhood', 'year', 'month'], inplace=True)

        df_submission.loc[df_output.index, :] = df_output.values

    if output_file is not None:
        logger.info(f"writing {len(df_submission):,} rows out to {output_file}")
        df_submission.to_csv(output_file, index=True)

    return df_submission

if __name__ == "__main__":
    typer.run(main)
