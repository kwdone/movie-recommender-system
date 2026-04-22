from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.preprocessing import normalize

# This will extract features from overviews and other data sources 
# to create the feature vector used for content-based filtering
class ContentAnalyzer:
    def __init__(self):
        self.tf_idf = None 
        self.vectorizer = TfidfVectorizer(stop_words='english',
                                          min_df=2,
                                          max_df=0.8, 
                                          max_features=10000)
        self.overviews = None
        self.item_features = None
        self.user_profile = None
        self.movie_id_to_index = {}
        self.index_to_movie_id = {}
        
    def build_mapping_idx(self, movie_ids):
        for idx, movie_id in enumerate(movie_ids):
            self.movie_id_to_index[movie_id] = idx
            self.index_to_movie_id[idx] = movie_id

    # fit() method assumes features arg to be a dict for multimodal data
    def fit(self, train_df, metadata):
        all_items = metadata["movie_id"].tolist()

        metadata["overview"] = metadata["overview"].fillna("")
        self.build_mapping_idx(all_items)
        self.tf_idf = self.vectorizer.fit_transform(metadata["overview"])
        self.user_profile = self.build_user_profiles(train_df)
        
        # Normalize to compute cosine similarity through dot product
        self.tf_idf = normalize(self.tf_idf)
        for user in self.user_profile:
            vec = self.user_profile[user]
            self.user_profile[user] = normalize(vec)
    
    def predict(self, user, item):
        if item not in self.movie_id_to_index:
            return None
        
        user_profile = self.user_profile[user]
        item_id = self.movie_id_to_index[item]
        item_representation = self.tf_idf[item_id]
        return user_profile @ item_representation.T

    def recommend(self, user_id, k, exclude_items=None):
        scores = {
            i: cosine_similarity(self.user_profile[user_id], self.item_vectors[i])
            for i in self.item_vectors
            if i not in exclude_items
        }
        return sorted(scores, key=scores.get, reverse=True)[:k]
    
    def build_user_profiles(self, ratings_df):
        user_profile = {}

        for user in ratings_df['user_id'].unique():
            user_data = ratings_df[ratings_df['user_id'] == user]

            item_indices = user_data['movie_id'].values
            ratings = user_data['rating'].values

            # Filter both item_indices and ratings together
            valid_indices = [i for i, movie_id in enumerate(item_indices) if movie_id in self.movie_id_to_index]
            item_indices = [self.movie_id_to_index[item_indices[i]] for i in valid_indices]
            ratings = ratings[valid_indices]

            if len(item_indices) == 0:
                continue
            
            item_vectors = self.tf_idf[item_indices]

            user_item_vectors = item_vectors.toarray()

            profile = (user_item_vectors.T @ ratings) / np.sum(ratings)
            user_profile[user] = profile.reshape(1, -1)
        
        return user_profile
