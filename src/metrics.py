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

    Returns
    -------
    float
        AP@K.
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
     preds : dict
         Model predictions as a list of tuples where the first element is the playlist id
         and the second element is the list of predictions for that playlist.
     targets : dict
         Target (true) values for every playlist.
     k : int, optional
         The K in the MAP@K.

     Returns
     -------
     float
        MAP@K.
     """

    ap = 0
    for playlist_id, values in targets.items():
        pred = preds[playlist_id]
        ap += ap_at_k(pred, values, k=min(len(values), k))

    return ap / len(targets)
