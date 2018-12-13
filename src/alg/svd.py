import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

from src.alg.recsys import RecSys
from src.alg.utils import knn


class SVD(RecSys):

    def __init__(self, factors=8, knn=np.inf):
        super().__init__()
        self.factors = factors
        self.knn = knn

    def compute_similarity(self, dataset):
        _, _, vt = svds(dataset.asfptype(), k=self.factors)
        vt = sp.csr_matrix(vt)
        s = vt.T * vt
        return s

    def rate(self, dataset, targets):
        s = self.compute_similarity(dataset)
        s = knn(s, self.knn)
        return dataset[targets, :] * s
