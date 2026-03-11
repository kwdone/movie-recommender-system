import numpy as np 

class ItemBasedCF:
    def __init__(self, k=5):
        self.k = k
        self.rating_matrix = None
        self.mask = None
        self.neighbor_idx = None 
        self.neighbor_sim = None
        self.similarity_matrix = None 

    def compute_similarity_matrix(self):
        # Compute the item-item similarity matrix using cosine similarity
        X = self.rating_matrix
        n_items = X.shape[1]
        sim = np.zeros((n_items, n_items))

        for i in range(n_items):
            for j in range(i, n_items):
                vec_i = X[:, i]
                vec_j = X[:, j]

                mask_ij = self.mask[:, i] & self.mask[:, j]

                if np.sum(mask_ij) == 0:
                    sim[i, j] = 0
                    continue
                
                v_i = vec_i[mask_ij]
                v_j = vec_j[mask_ij]

                numerator = np.dot(v_i, v_j)
                denominator = np.linalg.norm(v_i) * np.linalg.norm(v_j)
                
                value = numerator / denominator if denominator != 0 else 0 
                sim[j, i] = value 
                sim[i, j] = value
 
        return sim
    
    def fit(self, data, mask):
        self.rating_matrix = data
        self.mask = mask

        self.similarity_matrix = self.compute_similarity_matrix()

        n_items = self.similarity_matrix.shape[0]
        k = self.k

        neighbor_idx = np.zeros((n_items, k), dtype=np.int32)
        neighbor_sim = np.zeros((n_items, k), dtype=np.float32)

        for i in range(self.similarity_matrix.shape[0]):
            row = self.similarity_matrix[i].copy()
            row[i] = -np.inf
            top_k = np.argpartition(row, kth=-k)[-k:]
            top_k = top_k[np.argsort(row[top_k])[::-1]]
            
            neighbor_idx[i] = top_k
            neighbor_sim[i] = row[top_k]

        self.neighbor_idx = neighbor_idx 
        self.neighbor_sim = neighbor_sim

    def predict(self, user, item):
        numerator = 0.0
        denominator = 0.0
        # Predict a user's rating for an item
        sim_items = self.neighbor_idx[item]
        cos_sim = self.neighbor_sim[item]

        user_ratings = self.rating_matrix[user]

        for i in range(sim_items.shape[0]):
            sim = cos_sim[i]
            idx = sim_items[i]

            if self.mask[user, idx]:
                rating = user_ratings[idx] 

                numerator += rating * sim
                denominator += np.abs(sim)
        
        if denominator == 0:
            return 0

        return numerator / denominator

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

# if __name__ == "__main__":
#     data = np.array([[7, 6, 7, 4, 5, 4],
#                     [6, 7, 0, 4, 3, 4],
#                     [0, 3, 3, 1, 1, 0],
#                     [1, 2, 2, 3, 3, 4],
#                     [1, 0, 1, 2, 3, 3]])
    
#     data_mask, norm_data = normalize_matrix(data)
#     recommender = ItemBasedCF(norm_data, data_mask)
#     sim = recommender.similarity_matrix
#     print(sim)
            
