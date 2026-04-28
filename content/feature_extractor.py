from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.preprocessing import normalize
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse import hstack
import ast

# This will extract features from overviews and other data sources 
# to create the feature vector used for content-based filtering

class ContentAnalyzer:
    def __init__(self, genre_path,
                actor_vocab_path,
                director_vocab_path, 
                writer_vocab_path,
                w_genres=0.0098, w_tfidf=0.8583, w_director=0.4821, w_writer=0.4446, w_actor=0.6768):
        
        df = pd.read_csv(genre_path)
        self.actor_map = pd.read_csv(actor_vocab_path)
        self.director_map = pd.read_csv(director_vocab_path)
        self.writer_map = pd.read_csv(writer_vocab_path)

        # Collection of features
        self.tf_idf = None  # Shape (n_items, vocab_size) # CACHED 
        self.movie_genres = None # Shape (n_items, n_genres (20)) # CACHED
        self.actor_encoding = None
        self.writer_encoding = None
        self.director_encoding = None

        self.vectorizer = TfidfVectorizer(stop_words='english',
                                          min_df=2,
                                          max_df=0.8, 
                                          max_features=10000)
        
        self.overviews = None
        self.item_features = None
        self.user_profile = None 

        self.movie_id_to_index = {}
        self.index_to_movie_id = {}

        # Mapping dict for a genre's name to its corresponding column in the CSR matrix
        self.genre_to_id = dict(zip(df["genre"], df["id"]))
        
        # Mapping dict for actor's ID to the corresponding column in the CSR matrix
        self.actor_to_col = {}
        self.writer_to_col = {}
        self.director_to_col = {}

        self.user_id_to_row = {}
        self.row_to_user_id = {}

        self.w_genres = w_genres 
        self.w_tfidf = w_tfidf
        self.w_actors = w_actor
        self.w_directors = w_director
        self.w_writers = w_writer

    def build_mapping_idx(self, movie_ids):
        for idx, movie_id in enumerate(movie_ids):
            self.movie_id_to_index[movie_id] = idx
            self.index_to_movie_id[idx] = movie_id

        self.actor_to_col = {
            name: idx 
            for idx, name in enumerate(self.actor_map["actor"])
        }

        self.director_to_col = {
            name: idx 
            for idx, name in enumerate(self.director_map["director"])
        }

        self.writer_to_col = {
            name: idx 
            for idx, name in enumerate(self.writer_map["writer"])
        }

    def genre_process(self, x):
        if pd.isna(x):
            return []
        if isinstance(x, str):
            return ast.literal_eval(x)
        return x

    # fit() method assumes features arg to be a dict for multimodal data
    # feature vector = [TF-IDF | One-hot genres | ]
    def genre_encoding(self, metadata):
        rows, cols, data = [], [], []
        metadata = metadata.copy()
        metadata["genres_y"] = metadata["genres_y"].apply(self.genre_process)
        
        all_movies = metadata["movie_id"].tolist()
        all_movies_genres = metadata["genres_y"].tolist()
        for idx, genre_dict in enumerate(all_movies_genres):
            for genre in genre_dict:
                if genre["name"] not in self.genre_to_id:
                    continue

                genre = self.genre_to_id[genre["name"]]
                rows.append(idx)
                cols.append(genre)
                data.append(1)
        
        genre_encoding = coo_matrix((data, (rows, cols)), shape=(len(all_movies), len(self.genre_to_id)))
        genre_encoding = genre_encoding.tocsr()
        self.movie_genres = genre_encoding

    def cast_encoding(self, cast_metadata, metadata):
        cast_metadata = cast_metadata.fillna("")

        actor_rows, actor_cols, actor_data = [], [], []
        writer_rows, writer_cols, writer_data = [], [], []
        director_rows, director_cols, director_data = [], [], []

        cast_lookup = cast_metadata.set_index("movie_id")

        for idx, movie_id in enumerate(metadata["movie_id"]):

            if movie_id not in cast_lookup.index:
                continue

            row = cast_lookup.loc[movie_id]

            cast_val = row["cast"]
            writer_val = row["writers"]
            director_val = row["director"]

            cast_list = cast_val.split("|") if isinstance(cast_val, str) and cast_val else []
            writer_list = writer_val.split("|") if isinstance(writer_val, str) and writer_val else []
            director_list = director_val.split("|") if isinstance(director_val, str) and director_val else []

            for actor in cast_list:
                if actor in self.actor_to_col:
                    actor_rows.append(idx)
                    actor_cols.append(self.actor_to_col[actor])
                    actor_data.append(1)

            for writer in writer_list:
                if writer in self.writer_to_col:
                    writer_rows.append(idx)
                    writer_cols.append(self.writer_to_col[writer])
                    writer_data.append(1)

            for director in director_list:
                if director in self.director_to_col:
                    director_rows.append(idx)
                    director_cols.append(self.director_to_col[director])
                    director_data.append(1)

        self.actor_encoding = coo_matrix(
            (actor_data, (actor_rows, actor_cols)),
            shape=(len(metadata), len(self.actor_to_col))
        ).tocsr()

        self.writer_encoding = coo_matrix(
            (writer_data, (writer_rows, writer_cols)),
            shape=(len(metadata), len(self.writer_to_col))
        ).tocsr()

        self.director_encoding = coo_matrix(
            (director_data, (director_rows, director_cols)),
            shape=(len(metadata), len(self.director_to_col))
        ).tocsr()

    def fit(self, train_df, metadata, cast_metadata):
        all_items = metadata["movie_id"].tolist()

        metadata["overview"] = metadata["overview"].fillna("")

        self.build_mapping_idx(all_items)

        self.tf_idf = self.vectorizer.fit_transform(metadata["overview"])
        self.genre_encoding(metadata)
        self.cast_encoding(cast_metadata, metadata)

        self.tf_idf = normalize(self.tf_idf)
        self.movie_genres = normalize(self.movie_genres)

        self.item_features = hstack([
            self.w_tfidf * self.tf_idf,
            self.w_genres * self.movie_genres,
            self.w_actors * self.actor_encoding,
            self.w_writers * self.writer_encoding,
            self.w_directors * self.director_encoding
        ])

        self.item_features = normalize(self.item_features)

        self.build_user_profiles(train_df)
    
    def predict(self, user, item):
        if item not in self.movie_id_to_index:
            return None
        
        if user not in self.user_id_to_row:
            return None

        row = self.user_id_to_row[user]
        user_profile = self.user_profile[row]
        
        item_id = self.movie_id_to_index[item]

        item_representation = self.item_features[item_id]
        return user_profile @ item_representation.T
    
    def predict_many(self, user, items):
        row = self.user_id_to_row.get(user)
        if row is None:
            return []

        valid = [(i, self.movie_id_to_index[i]) for i in items if i in self.movie_id_to_index]

        if not valid:
            return []

        item_ids = [x[1] for x in valid]
        item_names = [x[0] for x in valid]

        scores = (self.user_profile[row] @ self.item_features[item_ids].T).toarray().ravel()

        return list(zip(item_names, scores))

    def recommend(self, user_id, k, exclude_items=None):
        row = self.user_id_to_row[user_id]
        scores = (self.user_profile[row] @ self.item_features.T).toarray().ravel()
        return scores
    
    def build_user_profiles(self, ratings_df):
        cats = ratings_df["user_id"].astype("category")

        users = cats.cat.codes
        self.user_id_to_row = {
            uid: i for i, uid in enumerate(cats.cat.categories)
        }

        items = ratings_df["movie_id"].map(self.movie_id_to_index)
        valid = items.notna()

        users = users[valid]
        items = items[valid].astype(int)
        ratings = ratings_df.loc[valid, "rating"]

        R = csr_matrix(
            (ratings, (users, items)),
            shape=(len(self.user_id_to_row), self.item_features.shape[0])
        )

        self.user_profile = normalize(R @ self.item_features)

