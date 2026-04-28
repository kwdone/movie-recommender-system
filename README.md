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

## 🚀 Implemented models

- Item-based Collaborative Filtering
- SVD for representation learning
- SVD for user–item latent factorization
- SVD++ using implicit feedback from users
- Content-based model on movies' metadata

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
| Content-based (TF-IDF only) | -	| 0.15308 |	0.0641 | 0.1 |
| Content-based (metadata provided)|  -  | 0.24744 | 0.0859 | 0.2154 |

---
## 📝 Key findings
### Collaborative Filtering
- Custom SVD performed competitively, slightly behind Surprise SVD 
- Incorporating user's implicit feedbacks improve metrics to near-library performance 
- Future gains likely from hyperparameter tuning and stronger ranking models

### Content-based Filtering
- TF-IDF baseline implementation achieved solid metrics, especially in high Recall@K, suggesting the model's capacity for providing broad relevant-item coverage
- Incorporating other metadata (genres, cast, director, writer) significantly improve all metrics
- Strong candidate for hybrid ensemble due to complementary signals.

---

## 💡 Tech stack
Python, pandas, sklearn, scipy, Optuna, 