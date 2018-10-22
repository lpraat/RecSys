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

    def run(self, targets, k=10):
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
        norms = dataset.sum(axis=0).A.ravel()
        norms_a = np.power(norms, self.alpha)
        if self.asym:
            assert 0. <= self.alpha <= 1., "alpha must be between 0 and 1"
            norms_b = np.power(norms, 1 - self.alpha)
            norm_factors = np.outer(norms_a, norms_b) + self.h
            del norms_b

        else:
            norm_factors = np.outer(norms_a, norms_a) + self.h

        del norms
        del norms_a

        norm_factors = np.divide(1, norm_factors, out=norm_factors, where=norm_factors != 0)

        # Update similarity matrix
        s = s.multiply(norm_factors).tocsr()

        # K-nearest neighbours
        if self.knn != np.inf:
            
            # For each row
            for row in range(len(s.indptr) - 1):

                # Get row start and end offsets
                row_start = s.indptr[row]
                row_end = s.indptr[row + 1]

                # Get data slice from row
                data = s.data[row_start:row_end]

                if len(data) > self.knn:
                    # Discard not meaningful data
                    # We take the smallest similarities in the data array
                    # and set those data values to 0 using row_start as offset
                    nn = np.argpartition(data, -self.knn)[:-self.knn]
                    s.data[nn + row_start] = 0

        # Apply qfunc
        if self.qfunc is not None:
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
            mask = 1 - dataset_i
            ratings_i = ratings_i * mask

            # Compute top k items
            top_idxs = np.argpartition(ratings_i, -k)[-k:]
            sorted_idxs = np.argsort(-ratings_i[top_idxs])
            pred = top_idxs[sorted_idxs]

            # Add prediction
            preds.append([i, list(pred)])

        print("elapsed: {:.3}s\n".format(timer() - start))
        del ratings

        # Return predictions
        return preds

    
    def evaluate(self, train_set="train_set", test_set="test_set", k=10):
        """ Evaluate model on train set using map metric """

        print("loading data ...")
        # Load data from cache
        train_set = self.cache.fetch(train_set)
        test_set = self.cache.fetch(test_set)
        assert train_set.shape[0] == len(test_set), "cardinality of train set and test set should match"

        print("computing similarity matrix ...")
        start = timer()
        # Compute similarity matrix
        s = cosine_similarity(train_set, alpha=self.alpha, asym=self.asym, h=self.h, knn=self.knn, dtype=np.float32)
        print("elapsed time: {:.3f}s\n".format(timer() - start))

        print("computing ratings matrix ...")
        start = timer()
        # Compute ratings matrix
        ratings = (train_set * s).tocsr()
        print("elapsed time: {:.3f}s\n".format(timer() - start))

        print("computing predictions ...")
        start = timer()
        # Get predictions
        preds = predict(ratings, targets=range(train_set.shape[0]), k=k, mask=train_set, invert_mask=True)
        print("elapsed time: {:.3f}s\n".format(timer() - start))

        print("evaluating model ...")
        # Evaluate model
        score = evaluate(preds, self.cache.fetch("test_set"))
        print("MAP@{}: {:.5f}\n".format(k, score))