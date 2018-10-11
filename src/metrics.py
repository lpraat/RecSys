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