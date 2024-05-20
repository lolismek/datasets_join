# imports
import pandas as pd
import networkx as nx
from tqdm import tqdm
from itertools import permutations
import benchmarks
import random
import utils
import sys
import csv

# constants
SAMPLES = 100 # how many random indices we choose to later measure accuracy

# reading data
google_df = pd.read_csv('data/google_dataset.csv', on_bad_lines = 'error', dtype = 'str')
facebook_df = pd.read_csv('data/facebook_dataset.csv', on_bad_lines = 'error', dtype = 'str')
website_df = pd.read_csv('data/website_dataset.csv', on_bad_lines = 'skip', dtype = 'str', sep = ';')

# preprocessing
google_df, facebook_df, website_df = utils.preprocess(google_df, facebook_df, website_df)


# extracting colomns of interest, in order of interest
important_cols = ['name', 'phone', 'category', 'city', 'region', 'country', 'address'] # ordered by priority
frames = [google_df[important_cols], facebook_df[important_cols], website_df[important_cols]] # TODO: try all permutations

# merging the datasets in all possible orders (3! = 6)
perms = permutations([0, 1, 2])

# keep a unique sets of random samples (across al permmutation) for benchmarking (see benchmarks.py)
random_indexes = []

# maximum accuracy achieved by one of the permutation, according to established benchmark 
max_accuracy = 0

# !!! note that accuracy and score are different

for _ in range(0, 6):

    # moving on to the next permutation
    perm = next(perms)

    # putting the dataframes in the permutations order
    current_frames = []
    for i in range(0, 3):
        current_frames.append(frames[perm[i]])

    # converting the data frames to simple python matrixes
    data_sets = []
    for i in range(0, 3):
        data_sets.append(current_frames[i].values.tolist())

    # merging first two data sets, and adding score to current scores
    current_score = 0
    data_sets[1], x = utils.match(data_sets[0], data_sets[1])
    current_score += x

    # merging the results of last operation with third data set, and adding score to current score
    data_sets[2], x = utils.match(data_sets[1], data_sets[2])
    current_score += x

    # removing very similar rows in resulted dataset
    data_sets[2] = utils.remove_similar_rows(data_sets[2])

    # if we have not yet generated the random indices for the samples with which we measure score
    # then we generate them now. we also keep the same indices for the next mergins, so that
    # score makes sense in the context of different permutations for merging
    if _ == 0:
        random_indexes = [random.randint(0, len(data_sets[2]) - 1) for _ in range(SAMPLES)]

    # measuring accuracy of merged dataset
    current_accuracy = benchmarks.check_accuracy(data_sets[2], random_indexes)

    # print score and accuracy
    print(current_score)
    print(current_accuracy)

    # if accuracy has improved, override previous merged dataset, as this one is better
    if current_accuracy > max_accuracy:
        max_accuracy = current_accuracy

        with open('assets/data.csv', 'w', newline='') as f:
            writer = csv.writer(f)

            writer.writerow(important_cols)
            writer.writerows(data_sets[2])