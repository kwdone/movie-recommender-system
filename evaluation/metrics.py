import random
import numpy as np
from collections import defaultdict

def compute_popular_items_weighted(df, top_n=None):
    item_score = defaultdict(float)

    for u, i, r in df[["user_id", "movie_id", "rating"]].itertuples(index=False, name=None):
        item_score[i] += r

    popular = sorted(item_score.items(), key=lambda x: x[1], reverse=True)

    return set([i for i, _ in popular[:top_n]]) if top_n else [i for i, _ in popular]

def safe_sample(pool, n):
    pool = list(pool)
    if len(pool) == 0:
        return []
    return random.sample(pool, min(n, len(pool)))

def get_scores(model, user, item):
    pred = model.predict(user, item)
    return pred.est if hasattr(pred, "est") else pred


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

def mae(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

import random

def measures_at_k(model, test_dict, train_dict, all_items, popular_items, k):
    precisions = []
    recalls = []

    for user in test_dict:
        relevant_items = {i for (i, r) in test_dict[user] if r >= 4}
        if not relevant_items:
            continue

        seen_items = {i for (i, r) in train_dict[user]}

        # Candidate sampling
        negatives = set(random.sample(list(all_items - seen_items - relevant_items), 200))
        popular = set(safe_sample(popular_items - seen_items, 100))

        candidates = relevant_items | negatives | popular
        candidates = list(candidates)

        # Score
        # scores = [
        #     (item, score)
        #     for item in candidates
        #     if (score := get_scores(model, user, item)) is not None
        # ]

        if hasattr(model, "predict_many"):
            scores = model.predict_many(user, candidates)
        else:
            scores = [(i, get_scores(model,user,i)) for i in candidates]

        scores.sort(key=lambda x: x[1], reverse=True)

        recommended = [i for i, _ in scores[:k]]

        hits = sum(1 for item in recommended if item in relevant_items)

        precisions.append(hits / k)
        recalls.append(hits / len(relevant_items))

    precision_at_k = sum(precisions) / len(precisions) if precisions else 0
    recall_at_k = sum(recalls) / len(recalls) if recalls else 0

    return precision_at_k, recall_at_k

def ndcg(model, test_dict, train_dict, all_items, popular_items, k=5):
    ndcg_scores = []

    test_dict_map = {
        user: dict(items)
        for user, items in test_dict.items()
    }

    for user in test_dict:
        relevant_items = {i for (i, r) in test_dict[user] if r >= 4}
        if not relevant_items:
            continue

        seen_items = {i for (i, _) in train_dict[user]}

        # candidate sampling
        negatives = set(random.sample(list(all_items - seen_items - relevant_items), 200))
        popular = set(safe_sample(popular_items - seen_items, 100))

        candidates = relevant_items | negatives | popular
        candidates = list(candidates)

        if hasattr(model, "predict_many"):
            scores = model.predict_many(user, candidates)
        else:
            scores = [(i, get_scores(model,user,i)) for i in candidates]

        scores.sort(key=lambda x: x[1], reverse=True)
        recommended = [i for i, _ in scores[:k]]

        # DCG
        dcg = sum(
            (2 ** test_dict_map[user].get(item, 0) - 1 if test_dict_map[user].get(item, 0) >= 4 else 0)
            / np.log2(i + 2)
            for i, item in enumerate(recommended)
        )

        # IDCG
        ideal_gains = sorted(
            [2 ** r - 1 for _, r in test_dict[user] if r >= 4],
            reverse=True
        )[:k]

        dcg_star = sum(
            r / np.log2(i + 2)
            for i, r in enumerate(ideal_gains)
        )

        if dcg_star == 0:
            continue

        ndcg_scores.append(dcg / dcg_star)

    return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0
    







