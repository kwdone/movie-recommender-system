import numpy as np 
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity

class SVDRecommender:
    def __init__(self, d=30):
        self.truncated_svd = None
        self.rating_matrix = None 
        self.embed_matrix = None
        self.neighbor_idx = None 
        self.neighbor_sim = None
        self.mask = None
        self.U = None 
        self.S = None 
        self.Vt = None
        self.d = d
        self.sqrtS = None
        self.user_embed = None
        self.item_embed = None

    def truncate_svd(self, U, S, Vt):
        U = U[:, :self.d] # Slicing the first 50 columns
        S = S[:self.d]
        Vt = Vt[:self.d, :]
        return U, S, Vt
    
    def fit(self, data, mask=None):
        n_items = data.shape[1]

        self.rating_matrix = data
        if mask is None:
            self.mask = data != 0
        else:
            self.mask = mask
        self.embed_matrix = np.zeros((n_items, self.d))

        U, S, Vt = np.linalg.svd(data, full_matrices=False)
        self.U, self.S, self.Vt = self.truncate_svd(U, S, Vt)

        self.embed_matrix = self.Vt.T * self.S
        self.sqrtS = np.sqrt(self.S)

        self.user_embed = self.U * self.sqrtS
        self.item_embed = self.Vt.T * self.sqrtS

    def compute_sim_matrix(self):
        X = self.embed_matrix
        
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        
        norms[norms == 0] = 1

        X_norm = X / norms 

        sim_matrix = X_norm @ X_norm.T

        return sim_matrix
    
    def compute_peer_groups(self, k=10):
        sim_matrix = self.compute_sim_matrix()
        n_items = sim_matrix.shape[0]
        neighbor_idx = np.zeros((n_items, k), dtype=np.int32)
        neighbor_sim = np.zeros((n_items, k), dtype=np.float32)
        for i in range(sim_matrix.shape[0]):
            row = sim_matrix[i].copy()
            row[i] = -np.inf
            top_k = np.argpartition(row, kth=-k)[-k:]
            top_k = top_k[np.argsort(row[top_k])[::-1]]
            
            neighbor_idx[i] = top_k
            neighbor_sim[i] = row[top_k]

        self.neighbor_idx = neighbor_idx 
        self.neighbor_sim = neighbor_sim


    def predict(self, user, item):
        return self.user_embed[user] @ self.item_embed[item]

    def recommend(self, user, n_items=10):
        rated_items = self.mask[user]
        unrated_items = np.where(~rated_items)[0]

        predictions = []
        for item in unrated_items:
            pred = self.predict(user, item)
            predictions.append((item, pred))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        top_items = [item for item, _ in predictions[:n_items]]

        return top_items

def normalize_matrix(data, mask):
    user_mean = np.sum(data * mask, axis=1) / np.sum(mask, axis=1)

    user_mean = user_mean.reshape(-1, 1)

    normalized = np.where(mask, data - user_mean, 0)

    return normalized, user_mean


if __name__ == "__main__":
    ratings = pd.read_csv("data/ml-1m/ratings.dat",
                          sep="::",
                          engine="python",
                          names=["user_id", "movie_id", "rating", "timestamp"])
    
    ratings = ratings.drop("timestamp", axis=1)
    R = ratings.pivot(index="user_id", columns="movie_id", values="rating").fillna(0)

    R = R.to_numpy()

    print(R.shape)

    U, S, Vt = np.linalg.svd(R)
    print(U.shape)
    print(S.shape)
    print(Vt.shape)
    print(S[:10])
    U = U[:, :50] # Slicing the first 50 columns
    S = S[:50]
    Vt = Vt[:50, :]
    embed_matrix = Vt.T * S
    print(f"Embedding matrix shape: {embed_matrix.shape}")
    print(embed_matrix[:10, :])
