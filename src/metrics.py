import numpy as np


def ap_at_k(pred, target, k=10):
    hits = 0
    acc = 0
    for i in range(k):

        if pred[i] in target:
            hits += 1
            acc += hits / (i + 1)

    return acc / min(k, len(target))


def map_at_k(preds, targets, k=10):
    """
    MAP(Mean Average Precision)@k
    This is the metric used in the competition to evaluate a submission.
    """
    return np.mean([ap_at_k(pred, target, k=k) for pred, target in zip(preds, targets)])


def leave_one_out(preds, test_set, k = 10):
    """
    Given the predictions and a test set, evaluates the performance
    based on the number of playlists whose test song is listed
    in the first k predictions
    """

    score = 0

    for i, pred in enumerate(preds):
        if test_set[i] in pred[:k]:
            score += 1.
    
    return score / len(preds)
