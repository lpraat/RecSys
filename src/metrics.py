"""
This file contains the metrics through which recommender models are evaluated.
"""


def ap_at_k(preds, targets, k=10):
    """
    Calculates the AP@K(Average Precision at K).

    Parameters
    -------------
    preds : list
        Predictions.
    targets : list
        Target (true) values.
    k : int, optional
        K in the MAP@K.
    """
    hits = 0.
    acc = 0.
    for i in range(k):

        if preds[i] in targets:
            hits += 1.0
            acc += hits / (i + 1)

    return acc / min(k, len(targets))


def evaluate(preds, targets, k=10):
    """
     Evaluates a model predictions against targets using MAP@K(Mean Average Precision at K) metric.

     Parameters
     -------------
     preds : list
         Model predictions as a list of tuples where the first element is the playlist id
         and the second element is the list of predictions for that playlist.
     targets : list
         Target (true) values for every playlist.
     k : int, optional
         The K in the MAP@K.
     """

    ap = 0
    for playlist_id, values in targets:
        pred = preds[playlist_id][1]
        ap += ap_at_k(pred, values, k=min(len(values), k))

    return ap / len(targets)


def leave_one_out(preds, test_set, k=10):
    """
    Given the predictions and a test set, evaluates the performance
    based on the number of playlists whose test song is listed
    in the first k predictions
    """

    score = 0

    for i, pred in enumerate(preds):
        test_set_i = test_set[i]
        for test_elem in test_set_i:
            if test_elem in pred:
                score += 1

    return score / len(preds)
