# 🎬 Building a Recommender System from Scratch

This project focuses on implementing recommender systems using multiple approaches and deploying a real-time recommendation system.

---

## 📊 Dataset
- **MovieLens-1M**  
  https://grouplens.org/datasets/movielens/1m/

---

## 🚀 Stage 1: Basic Recommender Algorithms

Implemented models:
- Item-based Collaborative Filtering
- SVD for representation learning
- SVD for user–item latent factorization
- SVD++ using implicit feedback from users
---

## 📈 Evaluation Metrics
Models are evaluated using:
- **RMSE** (rating prediction)
- **Precision@K** (ranking quality)

---

## 🧪 Results

| Model | RMSE | Precision@5 |
|------|------|------------|
| Custom SVD | 0.89 | 0.0555 |
| Surprise SVD | 0.8738 | 0.0672 |
| Custom SVD++ | 0.8737 | 0.0637 |

---

## 📝 Observations
- The original custom SVD implementation is competitive but still slightly behind the optimized implementation from the Surprise library.
- Further improvements may include hyperparameter tuning, regularization, and advanced models.
- Incorportation of user's implicit feedbacks in the original SVD model helps boost the metrics to be on par with that from Surprise library

---

## 🚀 Stage 2: Implementing Content-based Filtering (CBRSs)