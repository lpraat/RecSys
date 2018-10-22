"""
Perform an item-based collaborative filtering algorithm
to determine the ranking of each item for each user
"""
import numpy as np
from timeit import default_timer as timer

from .recsys import RecSys
from .utils import cosine_similarity, predict
from src.metrics import evaluate


class UserKNN(RecSys):
    def __init__(self, dataset="train_set", alpha=0.5, asym=True, knn=np.inf, h=0, qfunc=None):
        # Super constructor
        super().__init__(dataset)

        # Initial values
        self.alpha = np.float32(alpha)
        self.asym = asym
        self.h = np.float32(h)
        self.qfunc = qfunc
        self.neighbours = neighbours

    def run(self, targets, k=10):
        """ Get predictions for dataset """

        print("loading data ...\n")
        # Fetch dataset
        dataset = self.cache.fetch(self.dataset).tocsr()

        # Determine targets
        if targets is None:
            targets = self.cache.fetch("targets")
            if targets is None:
                targets = range(dataset.shape[0])

        print("computing similarity matrix ...")
        start = timer()
        # Compute similarity matrix
        cosine_similarity(dataset.T, alpha=self.alpha, asym=self.asym, h=self.h, knn=self.knn, dtype=np.float32)
        print("elapsed: {:.3F}s\n".format(timer() - start))

        print("computing ratings matrix ...")
        start = timer()
        # Compute playlist-track ratings
        ratings = dataset * s
        print("elapsed: {:.3}s\n".format(timer() - start))
        del s

        print("predicting ...")
        dataset = dataset.T
        ratings = ratings.T
        start = timer()
        # Predict
        preds = []
        for i in targets:
            # Get rows
            dataset_i = dataset.getrow(i).A.ravel().astype(np.uint8)
            ratings_i = ratings.getrow(i).A.ravel().astype(np.float32)

            # Filter out existing items
            mask = 1 - dataset_i
            ratings_i = ratings_i * mask

            # Compute top k items
            top_idxs = np.argpartition(ratings_i, -k)[-k:]
            sorted_idxs = np.argsort(-ratings_i[top_idxs])
            pred = top_idxs[sorted_idxs]

            # Add prediction
            preds.append([i, list(pred)])
            del dataset_i
            del ratings_i
            del mask

        print("elapsed: {:.3}s\n".format(timer() - start))
        del ratings

        # Return predictions
        return preds

    
    def evaluate(self, train_set = None):


        # @todo
        # Evaluate model
        score = evaluate(preds, self.cache.fetch("test_set"))
        print("MAP@{}: {:.5}\n".format(k, score))