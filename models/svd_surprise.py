from surprise import SVD as SurpriseSVD
from surprise import Dataset, Reader 
import pandas as pd

class SurpriseSVDWrapper:
    def __init__(self):
        self.model = SurpriseSVD()
    def fit(self, train_data):
        df = pd.DataFrame(train_data, columns=["user", "item", "rating"])

        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df, reader)
        train_set = data.build_full_trainset()
        self.model.fit(train_set)

    def predict(self, user, item):
        return self.model.predict(user, item).est
    
    def recommend(self, user_id, k, exclude_items=None):
        # score all items
        scores = {
            i: self.model.predict(user_id, i)
            for i in self.all_items
            if i not in exclude_items
        }
        return sorted(scores, key=scores.get, reverse=True)[:k]
