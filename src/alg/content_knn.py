"""
Perform an item-based collaborative filtering algorithm
to determine the ranking of each item for each user
"""

from timeit import default_timer as timer

import numpy as np
import scipy.sparse as sp

from src.metrics import evaluate
from .recsys import RecSys
from .utils import cosine_similarity, predict, knn


class ContentKNN(RecSys):
    """ Content based recommender """


    def __init__(self, *features, knn=np.inf):
        """
        Constructor

        Parameters
        -----------
        *features : list
            Set of features in the form of (feature x items) sparse matrix.
            Each feature is a tuple with a sparse matrix (or string)
            a weight and a dict of configurations
        """

        # Super constructor
        super().__init__()

        # Initial values
        self.features = features
        self.knn = knn

    def compute_similarity(self, dataset):

        # Create similarity matrix
        s = sp.csr_matrix((dataset.shape[1], dataset.shape[1]), dtype=np.float32)

        # Get similarity for each feature
        i = 0
        for feature, feature_w, feature_config in self.features:

            # Get feature data
            feature = self.cache.fetch(feature) if isinstance(feature, str) else feature
            feature = feature.tocsr()

            # Get feature configuration
            feature_alpha = feature_config["alpha"] if "alpha" in feature_config else 0.5
            feature_asym = feature_config["asym"] if "asym" in feature_config else True
            feature_h = feature_config["h"] if "h" in feature_config else 0

            print("loading data for feature {} ...\n".format(i))
            # Fetch feature from cache
            feature = self.cache.fetch(feature).tocsr()

            if feature is not None:
                print("computing similarity matrix for feature {} ...".format(i))
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
                print("feature {} not found".format(i))

            # Next feature
            feature += 1

        print("computing similarity knn...")
        start = timer()
        s = knn(s, self.knn)
        print("elapsed: {:.3f}s\n".format(timer() - start))

        return s

    def rate(self, dataset):
        s = self.compute_similarity(dataset)
        print("computing ratings matrix ...")
        start = timer()
        # Compute ratings
        ratings = (dataset * s).tocsr()
        print("elapsed: {:.3f}s\n".format(timer() - start))

        del s

        return ratings
