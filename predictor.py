import pandas as pd 

from evaluation.metrics import rmse, mae, precision_at_k
from data.loader import train_test_split
from data.loader import build_eval_dict

from model_factory import get_model

def main():
    data_path = "/data/ml-1m/ml-1m/ratings.dat"
    model_name = "surprise_svd"
    model = get_model(model_name)

    df = pd.read_csv(data_path, 
                     sep="::",
                     engine="python",
                     names=["user_id", "movie_id", "rating", "timestamp"])
    
    df = df.drop("timestamp", axis=1)
    train_data, test_data = train_test_split(df)

    train_dict_eval, test_dict_eval, all_items = build_eval_dict(df)

    model.fit(train_data)

    print(f"Evaluating for {model_name}")
    y_true = []
    y_pred = []
    for u, i, r in test_data:
        y_true.append(r)
        y_pred.append(model.predict(u, i))
    
    print(f"RMSE: {rmse(y_true, y_pred)}")
    print(f"MAE: {mae(y_true, y_pred)}")

    precisionK = precision_at_k(model, test_dict_eval, train_dict_eval, all_items, 5)
    print(f"Precision@K: {precisionK}")

if __name__ == "__main__":
    main()

