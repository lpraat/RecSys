"""
Perform an item-based collaborative filtering algorithm
to determine the ranking of each item for each user
"""

import gc
import numpy as np
import scipy.sparse as sp
import time
from timeit import default_timer as timer

from src.alg.recsys import RecSys
from src.data import save_file
from src.metrics import evaluate


class ItemKNN(RecSys):


    def __init__(self, h = 0, alpha = 0.5):
        # Super constructor
        super().__init__()

        # Initial values
        self.dataset    = "train_set"
        self.h          = h
        self.alpha      = alpha

    
    def run(self, targets, k = 10):
        """  """

        # Fetch dataset
        dataset = self.cache.fetch(self.dataset).tocsr()

        print("computing similarity matrix ...")
        start = timer()
        # Compute similarity matrix
        s = dataset.T * dataset
        s = s.tocsr()

        # Compute norms
        norms           = dataset.sum(axis = 0).A.ravel()
        norms           = np.power(norms, self.alpha)
        norm_factors    = np.outer(norms, norms) + self.h
        norm_factors    = np.divide(1, norm_factors, out = np.zeros_like(norm_factors), where = norm_factors != 0)

        # Release memory
        del norms
        
        # Update similarity matrix
        start = timer()
        s = s.multiply(norm_factors).tocsc()
        print("elapsed: {:.3}s\n".format(timer() - start))

        # Release memory
        del norm_factors

        print("computing ratings matrix ...")
        start = timer()
        # Compute playlist-track ratings
        ratings = dataset * s
        print("elapsed: {:.3}s\n".format(timer() - start))

        # Release memory
        del s

        print("predicting ...")
        start = timer()
        # Predict
        preds = []
        for i in targets:
            # Get rows
            dataset_i = dataset.getrow(i).A.ravel()
            ratings_i = ratings.getrow(i).A.ravel()

            # Filter out existing items
            ratings_i = ratings_i * (1 - dataset_i)

            # Compute top k items
            top_idxs    = np.argpartition(ratings_i, -10)[-10:]
            sorted_idxs = np.argsort(ratings_i[top_idxs])
            pred        = top_idxs[sorted_idxs]

            # Add prediction
            preds.append((i, list(pred)))

        print("elapsed: {:.3}s\n".format(timer() - start))

        # Release memory
        del ratings

        # @debug
        # Evaluate predictions
        score = evaluate(preds, self.cache.fetch("test_set"))
        print(score)

        # Return predictions
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