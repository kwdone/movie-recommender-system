import numpy as np
import pandas as pd 
from collections import defaultdict
import random

# Load ratings in the form of DataFrame 
def load_ratings(path):
    ratings = pd.read_csv(path, 
                        sep="::",
                        engine="python",
                        names=["user_id", "movie_id", "rating", "timestamp"])
    ratings = ratings.drop("timestamp", axis=1)
    return ratings

# Divide ratings into the two sets: train and test in the form of list of tuples
# The test data split is done for evaluating models in rating prediction 
def train_test_split(ratings, test_ratio=0.2, seed=42):
    np.random.seed(seed)
    ratings_of_users = defaultdict(list)
    train_data = []
    test_data = []

    ratings = list(ratings[["user_id", "movie_id", "rating"]].itertuples(index=False, name=None))
    for u, i, r in ratings:
        ratings_of_users[u].append((u, i, r))

    for user, user_ratings in ratings_of_users.items():
        np.random.shuffle(user_ratings)
        n_ratings = len(user_ratings)

        if n_ratings == 1:
            train_data.extend(user_ratings)
            continue

        n_test = max(1, int(test_ratio * n_ratings))

        test = user_ratings[0:n_test]
        train = user_ratings[n_test:]

        train_data.extend(train)
        test_data.extend(test)

    return train_data, test_data

# Divide ratings into two sets in the form of list of tuples 
# The test data is drawn, keeping only relevant items
# - A rating >= 4 (relevant item)
# - 100 ratings that have not been seen (implicit negatives)
def build_eval_dict(ratings):
    all_items = set(ratings["movie_id"])
    train_dict_eval = defaultdict(list)
    test_dict_eval = defaultdict(list)

    ratings_of_users = defaultdict(list)
    ratings = list(ratings[["user_id", "movie_id", "rating"]].itertuples(index=False, name=None))

    for u, i, r in ratings:
        ratings_of_users[u].append((u, i, r))
    
    for user, user_ratings in ratings_of_users.items():
        positives = [x for x in user_ratings if x[2] >= 4]

        if len(positives) < 2:
            continue 
            
        test_size = max(1, int(0.2 * len(positives)))
        test = random.sample(positives, test_size)
        train = [x for x in user_ratings if x not in test]

        for u, i, r in train:
            train_dict_eval[u].append((i, r))
        
        # u, i, r = test
        # test_dict_eval[u].append((i, r))
        for u, i, r in test:
            test_dict_eval[u].append((i, r))
    
    return train_dict_eval, test_dict_eval, all_items


    


