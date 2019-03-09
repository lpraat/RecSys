from timeit import default_timer as timer

import numpy as np

from .recsys import RecSys
from .utils import cosine_similarity, knn
from sklearn.preprocessing import normalize


class ItemKNN(RecSys):
    """
    Item recommender.

    Recommends items to users based on the similarity between items
    """

    def __init__(self, features=None, alpha=0.5, asym=True, knn=np.inf, h=0, normalize=True):
        """
        Constructor

        Parameters
        ---------------
        features : list
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
        """
        super().__init__()
        self.alpha = np.float32(alpha)
        self.asym = asym
        self.h = np.float32(h)
        self.knn = knn
        self.features = features if features else []
        self.normalize = normalize

    def compute_similarity(self, dataset):
        print("computing similarity ...")
        start = timer()
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

            # Fetch feature from cache
            print("loading data for feature {} ...\n".format(feature_i))
            feature = self.cache.fetch(feature).tocsr() if isinstance(feature, str) else feature

            if feature is not None:
                print("computing similarity for feature {} ...".format(feature_i))
                start = timer()

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

        if self.features and self.normalize:
            # Normalize the weighted sum
            s = normalize(s, norm='l2', axis=1)

        print("elapsed: {:.3f}s\n".format(timer() - start))
        return s

    def rate(self, dataset, targets):
        s = self.compute_similarity(dataset)

        print("computing ratings ...")
        start = timer()
        ratings = (dataset[targets, :] * s).tocsr()
        print("elapsed: {:.3f}s\n".format(timer() - start))
        del s

        return ratings
