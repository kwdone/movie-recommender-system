import numpy as np 
import pandas as pd
import random 
from collections import defaultdict
from tqdm import tqdm
from evaluation.metrics import precision_at_k

class SVDModel:
    def __init__(self):
        self.ratings = None # A tuple list of tuple (user, item, rating)
        self.P = None # User latent vector. Shape (k, n_users)
        self.Q = None # Item latent vector. Shape (k, n_items)
        self.bu = None # User bias vector. Shape (n_users, 1)
        self.bi = None # Item bias vector. Shape (n_items, 1)
        self.m_u = 0.0 # Global mean ratinng
        self.k = 50 # Latent dimensions
        self.lr = 5e-3
        self._lambda = 2e-2
        self.num_epochs = 10

    def fit(self, train_data):
        self.ratings = train_data 
        num_users = max(u for u,_,_ in train_data) + 1
        num_items = max(i for _,i,_ in train_data) + 1

        self.m_u = sum(r for _, _, r in train_data) / len(train_data)
        self.P = np.random.normal(0, 0.1, (num_users, self.k))
        self.Q = np.random.normal(0, 0.1, (num_items, self.k))

        self.bu = np.zeros(num_users)
        self.bi = np.zeros(num_items)

        for epoch in range(self.num_epochs):
            random.shuffle(train_data)
            rmse = 0.0
            for (u, i, r_ui) in tqdm(train_data, desc="Training"):
                pred = self.m_u + self.bi[i] + self.bu[u] + np.dot(self.Q[i].T, self.P[u])
                error = r_ui - pred 
                rmse += error ** 2

                pu = self.P[u]
                qi = self.Q[i]

                self.bu[u] = self.bu[u] + self.lr * (error - self._lambda * self.bu[u])
                self.bi[i] = self.bi[i] + self.lr * (error - self._lambda * self.bi[i])
                self.Q[i] = self.Q[i] + self.lr * (error * self.P[u] - self._lambda * qi)
                self.P[u] = self.P[u] + self.lr * (error * self.Q[i] - self._lambda * pu)

            print(f"RMSE for epoch {epoch}: {np.sqrt(rmse / len(train_data))}")

    def predict(self, user, item):  
        return self.m_u + self.bi[item] + self.bu[user] + np.dot(self.Q[item].T, self.P[user])

    def recommend(self, user, n_items=10):
        return None