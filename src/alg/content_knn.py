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


class ContentKNN(RecSys):


    def __init__(self, dataset = "train_set", features = [], h = 3, alpha = 0.5):
        # Super constructor
        super().__init__(dataset)

        # Initial values
        self.features   = features
        self.h          = h
        self.alpha      = np.float32(alpha)

    
    def run(self, targets, k = 10):
        """  """

        # Assert at least one feature
        assert len(self.features) > 0

        # Fetch dataset
        dataset = self.cache.fetch(self.dataset).tocsr()

        # Create ratings matrix
        ratings = sp.csr_matrix(dataset.shape, dtype = np.float32)

        for feature_name, feature_w in self.features:

            print("loading data for feature '{}' ...\n".format(feature_name))
            # Fetch feature from cache
            feature = self.cache.fetch(feature_name).tocsr()

            print("computing similarity matrix for feature '{}' ...".format(feature_name))
            start = timer()

            # Compute norms
            norms           = feature.sum(axis = 0).A.ravel()
            norms           = np.power(norms, self.alpha)
            norm_factors    = np.outer(norms, norms) + self.h
            norm_factors    = np.divide(1, norm_factors, out = np.zeros_like(norm_factors), where = norm_factors != 0)
            del norms

            # Compute similarity matrix
            s = feature.T * feature
            s = s.multiply(norm_factors).tocsr()
            del norm_factors
            print("elapsed: {:.3}s\n".format(timer() - start))
            
            print("computing ratings matrix for feature '{}' ...".format(feature_name))
            start = timer()
            # Compute playlist-track ratings using similarity between tracks
            ratings += (dataset * s) * feature_w
            del s
            print("elapsed: {:.3}s\n".format(timer() - start))
        
        # Take average ratings
        ratings /= sum([f[1] for f in self.features])

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
            preds.append((i, list(pred)))

        print("elapsed: {:.3}s\n".format(timer() - start))
        del ratings

        # Return predictions
        return preds

    
    def evaluate(self, train_set = None):


        # @todo
        # Evaluate model
        score = evaluate(preds, self.cache.fetch("test_set"))
        print("MAP@{}: {:.5}\n".format(k, score))