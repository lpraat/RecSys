from timeit import default_timer as timer

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize

from src.alg.recsys import RecSys
from src.alg.utils import knn


class HybridSimilarity(RecSys):

    """
    HybridSimilarity recommender.

    Recommends items to users by combining similarity matrices from different
    models.
    """

    def __init__(self, *models, knn=np.inf, mode='item', normalize=True):
        super().__init__()
        self.models = list(models)
        self.knn = knn
        self.mode = mode
        self.normalize = normalize

    def compute_similarity(self, dataset):

        if not self.models:
            raise RuntimeError("You already called rate")

        # shape[1] since we just use this method with item similarities
        if self.mode == 'item':
            s = sp.csr_matrix((dataset.shape[1], dataset.shape[1]), dtype=np.float32)
        else:
            s = sp.csr_matrix((dataset.shape[0], dataset.shape[0]), dtype=np.float32)

        while self.models:
            model, w = self.models.pop()
            model_similarity = model.compute_similarity(dataset)
            model_similarity = model_similarity * w

            s += model_similarity
            del model_similarity

        if self.normalize:
            s = normalize(s, norm='l2', axis=1)
        s = knn(s, self.knn)
        return s

    def rate(self, dataset):
        s = self.compute_similarity(dataset)

        print("computing ratings ...")
        start = timer()
        if self.mode == 'item':
            ratings = (dataset * s).tocsr()
        else:
            ratings = (dataset.T * s).tocsr()
        del s
        print("elapsed: {:.3f}s\n".format(timer() - start))


        if self.mode == 'item':
            return ratings
        else:
            return ratings.T
