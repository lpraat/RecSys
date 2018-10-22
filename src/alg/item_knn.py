"""
Perform an item-based collaborative filtering algorithm
to determine the ranking of each item for each user
"""

import gc
import numpy as np
import scipy.sparse as sp
import time
from timeit import default_timer as timer

from .recsys import RecSys
from src.data import save_file
from src.metrics import evaluate


class ItemKNN(RecSys):


    def __init__(self, dataset = "train_set", alpha = 0.5, asym = True, h = 3, qfunc = None):
        # Super constructor
        super().__init__(dataset)

        # Initial values
        self.alpha  = np.float32(alpha)
        self.asym   = asym
        self.h      = np.float32(h)
        self.qfunc  = qfunc

    
    def run(self, targets, k = 10):
        """  """

        print("loading data ...\n")
        # Fetch dataset
        dataset = self.cache.fetch(self.dataset).tocsr()

        print("computing similarity matrix ...")
        start = timer()
        # Compute similarity matrix
        s = dataset.T * dataset
        s = s.tocsr()

        # Compute norms
        norms           = dataset.sum(axis = 0).A.ravel()
        norms_a         = np.power(norms, self.alpha)
        if self.asym:
            assert self.alpha >= 0. and self.alpha <= 1.
            norms_b         = np.power(norms, 1 - self.alpha)
            norm_factors    = np.outer(norms_a, norms_b) + self.h
        else:
            norm_factors    = np.outer(norms_a, norms_a) + self.h
        norm_factors    = np.divide(1, norm_factors, out = np.zeros_like(norm_factors), where = norm_factors != 0)
        del norms
        del norms_a
        del norms_b
        
        # Update similarity matrix
        start = timer()
        s = s.multiply(norm_factors).tocsr()

        # Apply qfunc
        if self.qfunc != None:
            qfunc = np.vectorize(self.qfunc)
            s.data = qfunc(s.data)
        del norm_factors
        print("elapsed: {:.3}s\n".format(timer() - start))

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
            dataset_i = dataset.getrow(i).A.ravel().astype(np.uint8)
            ratings_i = ratings.getrow(i).A.ravel().astype(np.float32)

            # Filter out existing items
            mask        = 1 - dataset_i
            ratings_i   = ratings_i * mask

            # Compute top k items
            top_idxs    = np.argpartition(ratings_i, -k)[-k:]
            sorted_idxs = np.argsort(-ratings_i[top_idxs])
            pred        = top_idxs[sorted_idxs]

            # Add prediction
            preds.append([i, list(pred)])

        print("elapsed: {:.3}s\n".format(timer() - start))

        # Release memory
        del ratings

        # @debug
        # Evaluate predictions
        score = evaluate(preds, self.cache.fetch("test_set"))
        print("MAP@{}: {:.5}\n".format(k, score))

        # Return predictions
        return preds