import random
import numpy as np

def get_scores(model, user, item):
    pred = model.predict(user, item)
    return pred.est if hasattr(pred, "est") else pred

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

def mae(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

def precision_at_k(model, test_dict, train_dict, all_items, k):
    precisions = []

    for user in test_dict:

        relevant_items = set(test_dict[user])
        if not relevant_items:
            continue

        seen_items = set(train_dict[user])

        negatives = random.sample(
            list(all_items - seen_items - relevant_items), 100
        )

        candidates = negatives + list(relevant_items)

        scores = [(item, get_scores(model, user, item)) for item in candidates]

        scores.sort(key=lambda x: x[1], reverse=True)

        recommended = [i for i, _ in scores[:k]]

        hits = sum(1 for item in recommended if item in relevant_items)

        precisions.append(hits / k)

    return sum(precisions) / len(precisions) if precisions else 0