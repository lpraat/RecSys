"""
Perform an item-based collaborative filtering algorithm
to determine the ranking of each item for each user
"""
import numpy as np
from timeit import default_timer as timer

from .recsys import RecSys
from .utils import cosine_similarity, predict
from src.metrics import evaluate


class ItemKNN(RecSys):
    """ Recommend system based on items similarity """


    def __init__(self, dataset="interactions", alpha=0.5, asym=True, knn=np.inf, h=0, qfunc=None):
        # Super constructor
        super().__init__(dataset)

        # Initial values
        self.alpha = np.float32(alpha)
        self.asym = asym
        self.h = np.float32(h)
        self.qfunc = qfunc
        self.knn = knn

    def run(self, targets=None, k=10):
        """  """

        print("loading data ...\n")
        # Fetch dataset
        dataset = self.cache.fetch(self.dataset).tocsr()

        # Determine targets
        if targets is None:
            targets = range(dataset.shape[0])
        else:
            targets = self.cache.fetch(targets)

        print("computing similarity matrix ...")
        start = timer()
        # Compute similarity matrix
        s = cosine_similarity(dataset, alpha=self.alpha, asym=self.asym, h=self.h, knn=self.knn, qfunc=self.qfunc, dtype=np.float32)
        print("elapsed: {:.3f}s\n".format(timer() - start))

        print("computing ratings matrix ...")
        start = timer()
        # Compute playlist-track ratings
        ratings = (dataset * s).tocsr()
        print("elapsed: {:.3f}s\n".format(timer() - start))
        del s

        print("predicting ...")
        start = timer()
        # Predict
        preds = predict(ratings, targets=targets, k=k, mask=dataset, invert_mask=True)
        print("elapsed: {:.3f}s\n".format(timer() - start))
        del ratings

        # Return predictions
        return preds

    
    def evaluate(self, train_set="train_set", test_set="test_set", k=10):
        """ Evaluate model on train set using MAP@k metric """

        print("loading data ...\n")
        # Load data from cache
        train_set = self.cache.fetch(train_set)
        test_set = self.cache.fetch(test_set)
        assert train_set.shape[0] == len(test_set), "cardinality of train set and test set should match"

        print("computing similarity matrix ...")
        start = timer()
        # Compute similarity matrix
        s = cosine_similarity(train_set, alpha=self.alpha, asym=self.asym, h=self.h, knn=self.knn, qfunc=self.qfunc, dtype=np.float32)
        print("elapsed time: {:.3f}s\n".format(timer() - start))

        print("computing ratings matrix ...")
        start = timer()
        # Compute ratings matrix
        ratings = (train_set * s).tocsr()
        print("elapsed time: {:.3f}s\n".format(timer() - start))
        del s

        print("computing predictions ...")
        start = timer()
        # Get predictions
        preds = predict(ratings, targets=range(train_set.shape[0]), k=k, mask=train_set, invert_mask=True)
        print("elapsed time: {:.3f}s\n".format(timer() - start))
        del ratings

        print("evaluating model ...")
        # Evaluate model
        score = evaluate(preds, test_set)
        print("MAP@{}: {:.5f}\n".format(k, score))

        return score