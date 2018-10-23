"""
Perform an item-based collaborative filtering algorithm
to determine the ranking of each item for each user
"""

from timeit import default_timer as timer

import numpy as np
import scipy.sparse as sp

from src.metrics import evaluate
from .recsys import RecSys
from .utils import cosine_similarity, predict


class ContentKNN(RecSys):
    def __init__(self, dataset="interactions", features=[]):
        # Super constructor
        super().__init__(dataset)

        # Initial values
        self.features = features

    def run(self, targets=None, k=10):
        """ Get k predictions for each user """

        # Assert at least one feature
        assert len(self.features) > 0, "no feature specified"

        # Fetch dataset
        dataset = self.cache.fetch(self.dataset).tocsr()

        # Determine targets
        if targets is None:
            targets = range(dataset.shape[0])
        else:
            targets = self.cache.fetch(targets)

        # Create similarity matrix
        s = sp.csr_matrix((dataset.shape[1], dataset.shape[1]), dtype=np.float32)

        # Get similarity for each feature
        for feature_name, feature_w, feature_config in self.features:

            # Get feature configuration
            try:
                feature_alpha = feature_config["alpha"]
            except KeyError:
                feature_alpha = 0.5
            try:
                feature_asym = feature_config["asym"]
            except KeyError:
                feature_asym = True
            try:
                feature_h = feature_config["h"]
            except KeyError:
                feature_h = 0
            try:
                feature_knn = feature_config["knn"]
            except KeyError:
                feature_knn = np.inf
            try:
                feature_qfunc = feature_config["qfunc"]
            except KeyError:
                feature_qfunc = None

            print("loading data for feature '{}' ...\n".format(feature_name))
            # Fetch feature from cache
            feature = self.cache.fetch(feature_name).tocsr()

            if feature is not None:
                print("computing similarity matrix for feature '{}' ...".format(feature_name))
                start = timer()
                # Compute similarity matrix
                s += cosine_similarity(
                    feature,
                    alpha=feature_alpha,
                    asym=feature_asym,
                    h=feature_h,
                    knn=feature_knn,
                    qfunc=feature_qfunc,
                    dtype=np.float32
                ) * feature_w
                print("elapsed: {:.3f}s\n".format(timer() - start))

            else:
                print("feature {} not found".format(feature_name))

        print("computing ratings matrix ...")
        start = timer()
        # Compute playlist-track ratings using similarity between tracks
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
        """ Evaluate model performance using MAP@k metric """

        # Assert at least one feature
        assert len(self.features) > 0, "no feature specified"

        print("loading data ...\n")
        # Load data from cache
        train_set = self.cache.fetch(train_set)
        test_set = self.cache.fetch(test_set)
        assert train_set.shape[0] == len(test_set), "cardinality of train set and test set should match"

        # Create similarity matrix
        s = sp.csr_matrix((train_set.shape[1], train_set.shape[1]), dtype=np.float32)

        # Get similarity for each feature
        for feature_name, feature_w, feature_config in self.features:

            # Get feature configuration
            try:
                feature_alpha = feature_config["alpha"]
            except KeyError:
                feature_alpha = 0.5
            try:
                feature_asym = feature_config["asym"]
            except KeyError:
                feature_asym = True
            try:
                feature_h = feature_config["h"]
            except KeyError:
                feature_h = 0
            try:
                feature_knn = feature_config["knn"]
            except KeyError:
                feature_knn = np.inf
            try:
                feature_qfunc = feature_config["qfunc"]
            except KeyError:
                feature_qfunc = None

            print("loading data for feature '{}' ...\n".format(feature_name))
            # Fetch feature from cache
            feature = self.cache.fetch(feature_name).tocsr()

            if feature is not None:
                print("computing similarity matrix for feature '{}' ...".format(feature_name))
                start = timer()
                # Compute similarity matrix
                s += cosine_similarity(
                    feature,
                    alpha=feature_alpha,
                    asym=feature_asym,
                    h=feature_h,
                    knn=feature_knn,
                    qfunc=feature_qfunc,
                    dtype=np.float32
                ) * feature_w
                print("elapsed: {:.3f}s\n".format(timer() - start))

            else:
                print("feature {} not found".format(feature_name))

        print("computing ratings matrix ...")
        start = timer()
        # Compute playlist-track ratings using similarity between tracks
        ratings = (train_set * s).tocsr()
        print("elapsed: {:.3f}s\n".format(timer() - start))
        del s

        print("predicting ...")
        start = timer()
        # Predict
        preds = predict(ratings, targets=range(train_set.shape[0]), k=k, mask=train_set, invert_mask=True)
        print("elapsed: {:.3f}s\n".format(timer() - start))
        del ratings

        # Evaluate model
        score = evaluate(preds, test_set)
        print("MAP@{}: {:.5}\n".format(k, score))

        return score
