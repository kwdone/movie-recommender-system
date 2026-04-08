import pandas as pd 
import numpy as np
from data_split import per_user_split
from item_cf import ItemBasedCF
from item_cf import normalize_matrix
from svd import SVDRecommender

from surprise import Reader, Dataset
from surprise import SVD, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy

# if __name__ == "__main__":
#     reader = Reader(rating_scale=(1, 5))
#     df = pd.read_csv("data/ml-1m/ratings.dat",
#                           sep="::",
#                           engine="python",
#                           names=["user_id", "movie_id", "rating", "timestamp"])
#     traindf, testdf = per_user_split(df, test_ratio=0.2)
#     # trainset, testset = train_test_split(data, test_size=0.2)

#     trainset = Dataset.load_from_df(
#         traindf[["user_id", "movie_id", "rating"]],
#         reader
#     )

#     trainset = trainset.build_full_trainset()

#     testset = list(testdf[["user_id","movie_id","rating"]].itertuples(index=False, name=None))

#     model = SVD()

#     model.fit(trainset)

#     predictions = model.test(testset)

#     print(f"RMSE: {accuracy.rmse(predictions)}")
#     print(f"MAE: {accuracy.mae(predictions)}")

if __name__ == "__main__":
    # Loading data from ratings.dat
    ratings = pd.read_csv("data/ml-1m/ratings.dat",
                          sep="::",
                          engine="python",
                          names=["user_id", "movie_id", "rating", "timestamp"])
    
    ratings = ratings.drop("timestamp", axis=1)

    # Splitting the original matrix into train and test matrix
    train_df, test_df = per_user_split(ratings)

    train_matrix = train_df.pivot(columns="movie_id", index="user_id", values="rating")
    test_matrix = test_df.pivot(columns="movie_id", index="user_id", values="rating")

    test_matrix = test_matrix.reindex(
        index=train_matrix.index,
        columns=train_matrix.columns)

    mask = train_matrix.notna().to_numpy()
    test_mask = test_matrix.notna().to_numpy()
    
    train_matrix = train_matrix.fillna(0).to_numpy()
    train_matrix, user_mean = normalize_matrix(train_matrix, mask)

    test_matrix = test_matrix.fillna(0).to_numpy()
    # Loading the model
    # model = ItemBasedCF()
    model = SVDRecommender()
    model.fit(train_matrix, mask=mask)
    model.compute_peer_groups()
    model.recommend(user=0)

    loss = 0
    count = 0

    test_users, test_items = np.where(test_mask)

    for u, i in zip(test_users, test_items):
        pred_norm = model.predict(u, i)

        prediction = pred_norm + user_mean[u]
        
        error = prediction - test_matrix[u, i]
        
        loss += error ** 2
        count += 1

    rmse = np.sqrt(loss / count)
    print("RMSE: ", rmse) 
