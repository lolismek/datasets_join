# imports
import utils
import pandas as pd
import utils
import random
from tqdm import tqdm

# This function return how accurate is the merged dataset (stored in marged_dataset).
# ds - the initial datasets appended together (trivial merging).

# It calculates the average similarity score of the rows in the merged dataset with 
# the rows of the trivial ds.

# Because of efficiency reasons, we only sample a couple of rows from merged dataset, 
# situated at random_indexes

def check_accuracy(merged_ds, random_indexes):
    # data loading similar to main.py (see explanations there)
    google_df = pd.read_csv('data/google_dataset.csv', on_bad_lines = 'skip', dtype='str')
    facebook_df = pd.read_csv('data/facebook_dataset.csv', on_bad_lines = 'skip', dtype='str')
    website_df = pd.read_csv('data/website_dataset.csv', on_bad_lines = 'skip', dtype = 'str', sep = ';')

    google_df, facebook_df, website_df = utils.preprocess(google_df, facebook_df, website_df)

    google_df, facebook_df, website_df = utils.preprocess(google_df, facebook_df, website_df)
    important_cols = ['name', 'phone', 'category', 'city', 'region', 'country', 'address'] # ordered by priority
    frames = [google_df[important_cols], facebook_df[important_cols], website_df[important_cols]] # TODO: try all permutations

    # trivial appending of datasets
    ds = []
    for i in range(0, 3):
        tmp = frames[i].values.tolist()
        for j in range(0, len(tmp)):
            for k in range(0, len(tmp[j])):
                tmp[j][k] = str(tmp[j][k])
            ds.append(tmp[j])

    # getting the sum of all score
    total_score = 0
    for index in tqdm(random_indexes, desc = 'measuring accuracy score'):
        for i in range(0, len(ds)):
            # note that we only use the non-exponential version of the function for ACCURACY
            # !!! (this is not mandatory when calculating SCORE)
            total_score += utils.get_score(ds[i], merged_ds[index], 0)

    # take the average and return it
    return total_score / (len(random_indexes) * len(ds))