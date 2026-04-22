import numpy as np 
import pandas as pd
import random 
from collections import defaultdict
from tqdm import tqdm

class SVDModel:
    def __init__(self):
        self.ratings = None # A tuple list of tuple (user, item, rating)
        self.P = None # User latent vector. Shape (k, n_users)
        self.Q = None # Item latent vector. Shape (k, n_items)
        self.bu = None # User bias vector. Shape (n_users, 1)
        self.bi = None # Item bias vector. Shape (n_items, 1)
        self.user_implicit = None

        self.Y = None # Second item latent factors. Shape (k, n_items)
        self.per_user_ratings = None
        self.m_u = 0.0 # Global mean ratinng
        self.k = 50 # Latent dimensions
        self.lr = 7e-3
        self._lambda = 5e-3
        self._beta = 15e-3
        self.num_epochs = 20

    def fit(self, train_data):
        self.ratings = train_data 
        self.per_user_ratings = defaultdict(list)
        self.user_implicit = defaultdict(list)

        for (u, i, r_ui) in train_data:
            self.per_user_ratings[u].append(i)

        num_users = max(u for u,_,_ in train_data) + 1
        num_items = max(i for _,i,_ in train_data) + 1

        self.m_u = sum(r for _, _, r in train_data) / len(train_data)
        self.P = np.random.normal(0, 0.1, (num_users, self.k))
        self.Q = np.random.normal(0, 0.1, (num_items, self.k))
        self.Y = np.random.normal(0, 0.01, (num_items, self.k))

        self.bu = np.zeros(num_users)
        self.bi = np.zeros(num_items)

        for epoch in range(self.num_epochs):
            random.shuffle(train_data)
            rmse = 0.0
            for (u, i, r_ui) in tqdm(train_data, desc="Training"):
                Nu = self.per_user_ratings[u]
                sample_size = min(50, len(Nu))
                sampled_Nu = random.sample(Nu, sample_size)

                sqrt_Nu = np.sqrt(len(sampled_Nu))
                implicit = self.Y[sampled_Nu].sum(axis=0) / sqrt_Nu

                pu = self.P[u].copy()
                qi = self.Q[i].copy()

                pred = self.m_u + self.bi[i] + self.bu[u] + np.dot(qi, pu + implicit)
                error = r_ui - pred 
                rmse += error ** 2

                self.bu[u] = self.bu[u] + self.lr * (error - self._lambda * self.bu[u])
                self.bi[i] = self.bi[i] + self.lr * (error - self._lambda * self.bi[i])

                self.Q[i] = self.Q[i] + self.lr * (error * (pu + implicit) - self._beta * qi)
                self.P[u] = self.P[u] + self.lr * (error * qi - self._beta * pu)
                
                for j in sampled_Nu:
                    self.Y[j] = self.Y[j] + self.lr * (error * qi / sqrt_Nu - self._beta * self.Y[j])

            print(f"RMSE for epoch {epoch}: {np.sqrt(rmse / len(train_data))}")

        for u in self.per_user_ratings:
            self.user_implicit[u] = sum(self.Y[j] for j in self.per_user_ratings[u]) / np.sqrt(len(self.per_user_ratings[u]))

    def predict(self, user, item):  
        return self.m_u + self.bi[item] + self.bu[user] + np.dot(self.Q[item].T, self.P[user])

    def recommend(self, user_id, k, exclude_items=None):
        # score all items
        scores = {
            i: self.model.predict(user_id, i)
            for i in self.all_items
            if i not in exclude_items
        }
        return sorted(scores, key=scores.get, reverse=True)[:k]