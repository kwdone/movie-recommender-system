import pandas as pd 

from evaluation.metrics import rmse, mae, measures_at_k, compute_popular_items_weighted, ndcg
from data.loader import train_test_split
from data.loader import build_eval_dict

from model_factory import get_model

EVAL_MODES = {
    "rating": False, # Evaluating model in rating prediction task, using RMSE, MAE
    "ranking": True # Evaluating model in recommendation task, using Precision@K
}

def evaluate_rating(model, test_data):
    y_true, y_pred = [], []

    for u, i, r in test_data:
        y_true.append(r)
        y_pred.append(model.predict(u, i))

    print(f"RMSE: {rmse(y_true, y_pred)}")
    print(f"MAE: {mae(y_true, y_pred)}")

def evaluate_ranking(model, test_dict, train_dict, all_items, popular_items, k=5):
    precisionK, recallK = measures_at_k(model, test_dict, train_dict, all_items, popular_items, k)
    NDCG = ndcg(model, test_dict, train_dict, all_items, popular_items, k)
    print(f"Precision@K: {precisionK}")
    print(f"Recall@K: {recallK}")
    print(f"NDCG@5: {NDCG}")

def main():
    data_path = "data/ml-1m/ml-1m/ratings.dat"
    metadata_path = "data/movielens_metadata.csv"

    model_name = "svd_model"
    model, model_type = get_model(model_name)

    df = pd.read_csv(data_path, 
                     sep="::",
                     engine="python",
                     names=["user_id", "movie_id", "rating", "timestamp"])
    
    df = df.drop("timestamp", axis=1)

    metadata = pd.read_csv(metadata_path)

    metadata = metadata.dropna(subset=["overview"])
    all_items = metadata["movie_id"].tolist()
    popular_items = compute_popular_items_weighted(df, top_n=200)

    print(f"Evaluating for {model_name}")
    if EVAL_MODES["rating"]:
        train_data, test_data = train_test_split(df)
        model.fit(train_data)
        evaluate_rating(model, test_data)

    if EVAL_MODES["ranking"]:
        if model_type == "cf":
            train_dict_eval, test_dict_eval, all_items = build_eval_dict(df)
        
            flat_list = [
                (u, i, r)
                for u, items in train_dict_eval.items()
                for (i, r) in items
            ]

            model.fit(flat_list)
            evaluate_ranking(model, test_dict_eval, train_dict_eval, all_items, popular_items)
        
        if model_type == "cb":
            train_dict_eval, test_dict_eval, all_items = build_eval_dict(df)
        
            flat_list = [
                (u, i, r)
                for u, items in train_dict_eval.items()
                for (i, r) in items
            ]

            train_df = pd.DataFrame(flat_list, columns=["user_id", "movie_id", "rating"])
            model.fit(train_df, metadata)
            evaluate_ranking(model, test_dict_eval, train_dict_eval, all_items, popular_items)
            

if __name__ == "__main__":
    main()

