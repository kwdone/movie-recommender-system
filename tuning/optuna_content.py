import optuna
from content.feature_extractor import ContentAnalyzer
import pandas as pd 
from data.loader import build_eval_dict
from evaluation.metrics import ndcg, compute_popular_items_weighted

metadata_path = "data/movielens_metadata.csv"
cast_path = "data/cast_and_crew.csv"
genre_path = "data/genres.csv"
metadata = pd.read_csv(metadata_path)
cast_metadata = pd.read_csv(cast_path)

data_path = "data/ml-1m/ml-1m/ratings.dat"

actor_vocab_path = "data/actor_vocabulary.csv"
director_vocab_path = "data/director_vocabulary.csv"
writer_vocab_path = "data/writer_vocabulary.csv"

df = pd.read_csv(data_path, 
                    sep="::",
                    engine="python",
                    names=["user_id", "movie_id", "rating", "timestamp"])

df = df.drop("timestamp", axis=1)

train_dict_eval, test_dict_eval, all_items = build_eval_dict(df)
popular_items = compute_popular_items_weighted(df, top_n=200)

flat_list = [
    (u, i, r)
    for u, items in train_dict_eval.items()
    for (i, r) in items
]

train_df = pd.DataFrame(flat_list, columns=["user_id", "movie_id", "rating"])
 
def build_model(params, genre_path):
    model = ContentAnalyzer(genre_path=genre_path, 
                            actor_vocab_path=actor_vocab_path,
                            director_vocab_path=director_vocab_path,
                            writer_vocab_path=writer_vocab_path,
                            w_genres=params["w_genre"], 
                            w_tfidf=params["w_tfidf"],
                            w_director=params["w_director"],
                            w_writer=params["w_writer"],
                            w_actor=params["w_actor"])
    model.fit(train_df, metadata, cast_metadata)
    return model 

def objective(trial):
    params = {
        "w_genre": trial.suggest_float("w_genre", 0.0, 2.0),
        "w_tfidf": trial.suggest_float("w_tfidf", 0.0, 2.0),
        "w_director": trial.suggest_float("w_director", 0.0, 2.0),
        "w_writer": trial.suggest_float("w_writer", 0.0, 2.0),
        "w_actor": trial.suggest_float("w_actor", 0.0, 2.0)
    }

    model = build_model(params, genre_path)

    score = ndcg(model, test_dict_eval, train_dict_eval, all_items, popular_items, k=5)

    return score

    
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)