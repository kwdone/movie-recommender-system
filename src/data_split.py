import numpy as np 
import pandas as pd 

def per_user_split(df, test_ratio=0.2, seed=42):
    np.random.seed(seed)
    train_list = []
    test_list = []

    for user_id, group in df.groupby("user_id"):
        n_ratings = len(group)

        if len(group) == 1:
            train_list.append(group)
            continue 

        n_test = max(1, int(n_ratings * test_ratio))

        test_indices = np.random.choice(
            group.index,
            size=n_test, 
            replace=False
        )

        test = group.loc[test_indices]
        train = group.drop(test_indices)

        train_list.append(train)
        test_list.append(test)

    train_df = pd.concat(train_list)
    test_df = pd.concat(test_list)

    return train_df, test_df

