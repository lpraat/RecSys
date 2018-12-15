"""
This file contains the ItemKNN recommender which performs
an item-based collaborative filtering algorithm to determine the ranking
of each item for each user.
"""

from timeit import default_timer as timer

import numpy as np

from .recsys import RecSys
from .utils import cosine_similarity, knn


class ItemKNN(RecSys):
    """
    Item recommender.

    Recommends items based on the similarity between items
    """

    def __init__(self, *features, alpha=0.5, asym=True, knn=np.inf, h=0, qfunc=None, splus=False):
        """
        Constructor

        Parameters
        ---------------
        *features : list
            A set of additional features, in the form of (feature x item) sparse matrix
            Features are combined when computing the similarity matrix
            A tuple contains a sparse matrix (or a string), a weight and a dict of configurations
        alpha : scalar
            Norm used in cosine similarity
        asym : bool
            If true similarity matrix is no more symmetric
        knn : integer
            Limit influence to knn most similar items
        h : scalar
            Shrink term
        qfunc : lambda
            A function individually applied to similarities
        """
        # Super constructor
        super().__init__()

        # Initial values
        self.alpha = np.float32(alpha)
        self.asym = asym
        self.h = np.float32(h)
        self.qfunc = qfunc
        self.knn = knn
        self.features = features
        self.splus = splus

    def compute_similarity(self, dataset=None):
        print("computing similarity ...")
        start = timer()
        # Compute similarity matrix
        s = cosine_similarity(dataset, alpha=self.alpha, asym=self.asym, h=self.h, dtype=np.float32)
        print("elapsed: {:.3f}s\n".format(timer() - start))

        # Compute similarity for features
        feature_i = 0
        for feature, feature_w, feature_config in self.features:

            # Get feature data
            feature = self.cache.fetch(feature) if isinstance(feature, str) else feature
            feature = feature.tocsr()

            # Get feature configuration
            feature_alpha = feature_config["alpha"] if "alpha" in feature_config else 0.5
            feature_asym = feature_config["asym"] if "asym" in feature_config else True
            feature_h = feature_config["h"] if "h" in feature_config else 0

            print("loading data for feature {} ...\n".format(feature_i))
            # Fetch feature from cache
            feature = self.cache.fetch(feature).tocsr() if isinstance(feature, str) else feature

            if feature is not None:
                print("computing similarity for feature {} ...".format(feature_i))
                start = timer()
                # Compute similarity matrix
                s += cosine_similarity(
                    feature,
                    alpha=feature_alpha,
                    asym=feature_asym,
                    h=feature_h,
                    dtype=np.float32
                ) * feature_w
                print("elapsed: {:.3f}s\n".format(timer() - start))

            else:
                print("feature {} not found".format(feature_i))

            # Next feature
            feature_i += 1

        print("computing similarity knn...")
        start = timer()
        s = knn(s, self.knn)
        print("elapsed: {:.3f}s\n".format(timer() - start))

        return s

    def rate(self, dataset, targets):
        print("Using all dataset " + str(dataset.nnz))

        s = self.compute_similarity(dataset)
        print("computing ratings ...")
        start = timer()

        print(len(targets))
        # Compute playlist-track ratings
        ratings = (dataset[targets, :] * s).tocsr()
        print("elapsed: {:.3f}s\n".format(timer() - start))
        del s

        return ratings
