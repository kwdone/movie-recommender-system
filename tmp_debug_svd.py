import pandas as pd
import numpy as np
from src.svd import SVDRecommender, normalize_matrix
from src.data_split import per_user_split

ratings = pd.read_csv('data/ml-1m/ratings.dat', sep='::', engine='python', names=['user_id','movie_id','rating','timestamp'])
ratings = ratings.drop('timestamp', axis=1)
train_df, test_df = per_user_split(ratings)
train_matrix = train_df.pivot(columns='movie_id', index='user_id', values='rating')
test_matrix = test_df.pivot(columns='movie_id', index='user_id', values='rating')
test_matrix = test_matrix.reindex(index=train_matrix.index, columns=train_matrix.columns)
mask = train_matrix.notna().to_numpy()
test_mask = test_matrix.notna().to_numpy()
train_matrix_filled = train_matrix.fillna(0).to_numpy()
train_norm, user_mean = normalize_matrix(train_matrix_filled, mask)

for label, matrix, add_mean in [('raw', train_matrix_filled, False), ('norm', train_norm, True)]:
    model = SVDRecommender()
    model.fit(matrix, mask=mask)
    model.compute_peer_groups()
    loss = 0.0
    count = 0
    for u, i in zip(*np.where(test_mask)):
        pred_norm = model.predict(u, i)
        pred = pred_norm + user_mean[u] if add_mean else pred_norm
        error = pred - test_matrix.to_numpy()[u, i]
        loss += error**2
        count += 1
    print(label, 'RMSE', np.sqrt(loss/count))
