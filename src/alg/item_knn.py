"""
Perform an item-based collaborative filtering algorithm
to determine the ranking of each item for each user
"""

import numpy as np
import scipy.sparse as sp
import time

from src.alg.recsys import RecSys
from src.data import save_file
from src.metrics import evaluate


class ItemKNN(RecSys):


    def __init__(self, h = 5, alpha = 0.5):
        # Super constructor
        super().__init__()

        # Initial values
        self.dataset    = "train_set"
        self.h          = h
        self.alpha      = alpha

    
    def run(self, targets):
        """  """

        train_set       = self.cache.fetch(self.dataset)
        train_set_csr   = train_set.tocsr()

        # Compute dot products between tracks
        s = train_set_csr.T * train_set_csr

        # Compute tracks vector norms
        norms = np.power(np.sum(train_set, axis = 0), self.alpha)
        norms = np.asarray(norms).squeeze()

        d = np.ndarray(s.shape, dtype = np.float32)
        for ti in range(s.shape[0]):
            print(ti)
            for tj in range(s.shape[1]):
                d[ti, tj] = norms[ti] * norms[tj] + self.h

        # Calc normalized similarity matrix
        s = s / d

        # Get tracks scores
        scores = train_set_csr * s

        # Get predictions
        preds = []
        for pi in targets:
            print(pi)
            # Get rankings for this playlist
            # and filter out already added tracks
            mask    = 1 - train_set.getrow(pi).toarray()
            ranking = np.multiply(scores[pi, :], mask)
            top = (-ranking).argsort()[:10]

            preds.append([pi, ] + top.tolist())

        print("MAP@k=10 = {}".format(evaluate(preds, self.cache.fetch("test_set"))))
        return preds

def get_rankings(interactions, *contexts, weights = [1], normalize = [True]):
    """ Given one ore more interaction matrices generate a ranking matrix from the similarity of the items """

    # Compute the similarity matrix
    print("computing similarity matrix ...\n")

    similarity = interactions.transpose().tocsr().dot(interactions.tocsc()) * (weights[0] / (interactions.shape[1] if normalize[0] else 1))
    for i, context in enumerate(contexts):
        similarity += context.transpose().tocsr().dot(context.tocsc()) * (weights[i + 1] / (context.shape[1] if normalize[i + 1] else 1))
    
    # Compute the ranking matrix
    print("computing ranking matrix ...\n")

    rankings = interactions.tocsr().dot(similarity.tocsc())

    return rankings