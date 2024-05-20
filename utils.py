# imports
import pandas as pd
import networkx as nx
from tqdm import tqdm
import numpy as np
from collections import Counter
import math
import sys

# constants
# TODO: adjust THRESHOLD
THRESHOLD = 0 # minimum value for two rows to form an edge in the network
COLUMNS = 7 # how many important columns we have in our data
EXPONENTIAL = False # do we get the score of an edge based on the exponential formula (described in powerpoint)

E = [1, 3, 7, 20, 55, 148, 403] # e^i, (i = 0, ..., COLUMNS - 1), rounded to integers

def preprocess(google_df, facebook_df, website_df):
    # note that the collumn headers have been manually modified to match with eachother
    # note that \" has been replaces with "" (empirical discovery)
    # still have to fix website_dataset.csv

    # standardizing phone number format
    website_df['phone'] = website_df['phone'].str.replace('+', '')

    # adding address column for simplicty
    website_df['address'] = np.nan

    # converting everything to lower caps
    # google_df = google_df.applymap(lambda x : str(x).lower())
    # facebook_df = google_df.applymap(lambda x : str(x).lower())
    # website_df = google_df.applymap(lambda x : str(x).lower())

    # returning processed data
    return google_df, facebook_df, website_df


# two methods to measure the score of two rows from two different data sets:
# 1. non-exponential: just add how many values match from coresponding columns
# 2. exponential: if a column is of more interest (view the order of 'important_cols' array in main.py),
#                 then it adds/substract exponentially more value to the score, depending if the values match

# this is the score used to measure both local score, and accuracy later
# !!! note that accuracy only uses the non exponential version of this function
def get_score(vec1, vec2, exponential): 
    score = 0
    
    if exponential:
        for i in range(0, len(vec1)):
            if vec1[i] == vec2[i]:
                score += E[COLUMNS - i - 1]
            else:
                score -= E[COLUMNS -i - 1]
    else:
        for i in range(0, len(vec1)):
            score += vec1[i] == vec2[i]
    
    return score

# this functions merges a group of rows with the same 'name' in the first dataset
# with a group of rows with same 'name' in secon dataset

# - we add edges between two 'nodes' from different groups if their similarity score is bigger than
# a THRESHOLD
# - in order to do this, we use the max_flow_max_cost algorithm (for details see cp-algorithms.com)
# - note that because of library constraints, we actually perform max_flow_min_cost but with negative weights (equivalent, if we revers the sign in the end)

# the function return all matached pairs, the unmatched nodes from the first group, and the maximum score achieved by the algorithm

def match_group(group1, group2, info_group1, info_group2):
    # handling special case for efficieny reasons
    # (e.g. a name only appears in one of the groups => it's redundant to perform matching between empty group and non-empty groups)
    if len(group1) == 0:
        return [], [], 0 
    elif len(group2) == 0:
        return [], group1, 0
    elif isinstance(info_group1[0][0], float) and math.isnan(info_group1[0][0]):
        return [], group1, 0 

    # creating graph
    G = nx.DiGraph()

    # creating source and destination nodes 
    source = -1
    destination = len(group1) + len(group2)

    # indices of nodes coresponding to second group are offseted by the size of group1, 
    # in order for their indices to not overlap
    offset = len(group1)

    # actually adding nodes in the graph
    G.add_node(source)
    G.add_node(destination)
    for i in range(0, len(group1)):
        G.add_node(i)
    for i in range(0, len(group2)):
        G.add_node(i + offset)

    # trying to add edges between every pair
    for i in range(0, len(group1)):
        for j in range(0, len(group2)):

            # calculating similarity score
            score = get_score(info_group1[i], info_group2[j], EXPONENTIAL)

            # adding edge if it exceeds THRESHOLD
            if score > THRESHOLD:
                G.add_edge(i, offset + j, weight = -score, capacity = 1)

    # adding edges between regular nodes, the source and the desination nodes
    for i in range(0, len(group1)):
        G.add_edge(source, i, weight = 0, capacity = 1)
    for i in range(0, len(group2)):
        G.add_edge(i + offset, destination, weight = 0, capacity = 1)

    # performing the algorithm
    flow_dict = nx.max_flow_min_cost(G, source, destination)

    max_cost = 0
    matched = []
    unmatched = []
    for u in flow_dict:
        if u >= 0 and u < len(group1): # <=> if the node is in group1
            is_matched = False

            # check all adjecent nodes and searched for matched pair
            for v in G[u]:
                if flow_dict[u][v] > 0: # <=> if matched
                    is_matched = True
                    cost = G.get_edge_data(u, v)['weight'] 
                    max_cost -= cost # reverse the negative sign

                    # add to matched pairs, unoffset the node from second group
                    matched.append((group1[u], group2[v - offset])) 

            # self-explanatory
            if not is_matched:
                unmatched.append(group1[u])

    # return found values
    return matched, unmatched, max_cost


# this function merges two datasets
# we merge rows only if the 'name' fiels coresponds

# - if there are more rows with the same 'name' in the first data set, or the second,
#   we do a max-cost max-matching in the bipartite graph formed by the rows in the first dataset (in the left)
#   and the rows in the second dataset (in the right)

# - we add an edge between any two rows on different sides, if their similiraty is bigger than
#   the established THRESHOLD

# - after performing the algorithm, if two rows matched, then the one in the second dataset is replaced with 
#   the on in the first. moreover, if a row in the first dataset is not matched to any, then it is just 
#   appended in the second dataset

def match(ds1, ds2):

    # creating set of all names in both datasets
    names = set()

    # dictionary mapping: 'name' -> array of indices of rows with that 'name' in dataset 1 and 2
    name_indx1 = {}
    name_indx2 = {}

    # building the first dicionary
    for i in range(0, len(ds1)):
        x = str(ds1[i][0])
        #print("!! " + x)
        names.add(x)
        if not x in name_indx1:
            name_indx1[x] = []
        name_indx1[x].append(i)

    #building the second dictionary
    for i in range(0, len(ds2)):
        x = str(ds2[i][0])
        names.add(x)
        if not x in name_indx2:
            name_indx2[x] = []
        name_indx2[x].append(i)

    # the gole is to maximize the total_cost.
    # this is not a measure of accuracy, but rather server as the driving force of the algorithm
    total_cost = 0

    # iterating over all names
    for name in tqdm(names, desc = 'loading'):

        # indices of rows with 'name' in first and secon datasets
        group1 = []
        group2 = []

        # we also store those rows in this arrays, to easire acces information
        info_group1 = []
        info_group2 = []

        # building group1 and info_group1
        if name in name_indx1:
            group1 = name_indx1[name]
            for indx in group1:
                info_group1.append(ds1[indx])

        # building group2 and info_group2
        if name in name_indx2:
            group2 = name_indx2[name]
            for indx in group2:
                info_group2.append(ds2[indx])
        
        # matching the two groups
        # specifically, the algorithm maximizes max_cost, which server the goal of maximizing total_cost
        matched, unmatched, max_cost = match_group(group1, group2, info_group1, info_group2)
        total_cost += max_cost

        # all matched rows in the secondata set get replaced with their match in first dataste
        for matched_pair in matched:
            ds2[matched_pair[1]] = ds1[matched_pair[0]]

        # all unmatched rows in the first dataset just get appended to second dataset
        for unmatched_row in unmatched:
            ds2.append(ds1[unmatched_row])

        # !!! note that unmatched rows in second dataset will continue to exist in secon dataset

    return ds2, total_cost


# Helper function for 'remove_similar_rows'
def compress(batch):
    batch = list(zip(*batch)) # transpose of matrix for easier work

    most_frequent = [] # most common element on each line (column in original matrix) 
    for col in batch:
        count = Counter(col)
        most_frequent_element = count.most_common(1)[0][0]
        most_frequent.append(most_frequent_element)

    # return compressed version
    return most_frequent

# This function takes almost final merged dataset and further merges similar rows 
# between themselves. 

# - The function sorts the dataset's (ds) rows lexicographically and breakes it into continuous subsequences
# in which all the rows are very similar (meaning that any two rows from the subsequence must differ in at most one column)

# - Then, each batch is compressed into a single row. For each element, the majority element is taken as representative.
# This compression is then added into the new dataset. 

# !!! Accuracy score improves significantly because of this 'compression'

def remove_similar_rows(ds):
    # makins sure every element is in 'str' format
    for i in range(0, len(ds)):
        for j in range(0, len(ds[i])):
            ds[i][j] = str(ds[i][j])

    # sorting lexicographicaly, such that similar elements are near each other
    ds = sorted(ds)

    # the resulted dataset
    new_ds = []

    # current batch (acts like a buffer)
    batch = []

    # iterating through rows
    for indx in tqdm(range(0, len(ds)), desc = 'compressing'):
        if len(batch) == 0: # just append to empty batch
            batch.append(ds[indx])
        else: 

            # test if current row can be inserted into batch
            # (that means that it differs with no more than one column than any other row in the batch)
            similar_to_batch = True
            for i in range(0, len(batch)):
                if get_score(ds[indx], batch[i], 0) < COLUMNS - 1:
                    similar_to_batch = False

            if similar_to_batch == True:
                batch.append(ds[indx])
            else:
                # append the compression of the batch to the new dataset
                new_ds.append(compress(batch))

                # create a new current batch, containing just current row
                batch = [ds[indx]]

    # in case last batch was not compressed and appended
    new_ds.append(compress(batch))

    # return result
    return new_ds