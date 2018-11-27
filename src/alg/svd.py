from src.alg.recsys import RecSys
from scipy.sparse.linalg import svds

import scipy.sparse as sp
import numpy as np

class SVD(RecSys):

    def __init__(self, factors=8, knn=np.inf):
        super().__init__()
        self.factors = factors
        self.knn = knn

    def compute_similarity(self, dataset):
        print('Computing S _URM_SVD...')

        URM = sp.csr_matrix(dataset, dtype=float)
        _, _, vt = svds(URM, k=self.factors)
        v = vt.T
        s = np.dot(v, vt)
        return s

    def rate(self, dataset):
        s = self.compute_similarity(dataset)
        return (dataset * s)
