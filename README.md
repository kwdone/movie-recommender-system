# 🎬 Building a Recommender System from Scratch

This project focuses on implementing recommender systems using multiple approaches and deploying a real-time recommendation system.

---

## 📊 Dataset
- **MovieLens-1M**  
  https://grouplens.org/datasets/movielens/1m/
- **TMDb Movie Dataset**
  https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset
- **movielens_metadata.csv**
  https://www.kaggle.com/datasets/kwahnguyen/movielens1m-metadata/data
---

## 🚀 Stage 1: Basic Recommender Algorithms

Implemented models:
- Item-based Collaborative Filtering
- SVD for representation learning
- SVD for user–item latent factorization
- SVD++ using implicit feedback from users

--- 

## 🚀 Stage 2: Implementing Content-based Filtering (CBRSs)
- Current implementation uses TF-IDF of movie overviews collected from the TMBd datasets as feature for content-based filtering. Future implemetation may use other such features as: genres, cast, release_date,... to enhance perormance

- This update batch also introduce other metrics to comprehensively assess the models (Recall@K and NDCG@K)

---

## 📈 Evaluation Metrics
Models are evaluated using:
- **RMSE** (rating prediction)
- **Precision@K** (ranking quality)
- **NDCG@K** (order of ranking quality)

---

## 🧪 Results

| Model | RMSE | Precision@5 | Recall@5 | NDCG@5
|------|------|------------|--------|--------|
| Surprise SVD | 0.8738 | 0.179161 | 0.0556 | 0.1696 |
| Custom SVD++ | 0.8737 | 0.17671 | 0.0542 | 0.1646 |
| Content-based |  -  | 0.15308 | 0.0641 | 0.1 |

---

## 📝 Observations
### Stage 1 observations: Collaborative performance
- The original custom SVD implementation is competitive but still slightly behind the optimized implementation from the Surprise library.
- Further improvements may include hyperparameter tuning, regularization, and advanced models.
- Incorportation of user's implicit feedbacks in the original SVD model helps boost the metrics to be on par with that from Surprise library

### Stage 2 observations: Content-based and its potential
- Content-based filtering achieved decent metrics for basic feature (only TF-IDF). Noticeably, the model's Recall@K is better than collaborative filtering's, suggesting its better coverage of all relevant items to users. 

- The low NDCG@K metric is expected as content-based filtering optimized its ranking based on pure vector similarity, putting less emphasis on the order of ranking compared to collaborative filtering

- Further tuning and feature engineering may help the content-based model catch up with collaborative filtering methods and potentially contribute to the final hybrid model to capture signals different from CF. 

---