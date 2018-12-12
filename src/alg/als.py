import numpy as np
import scipy.sparse as sp

from src.alg.recsys import RecSys

from implicit.als import AlternatingLeastSquares as als


class ALS(RecSys):

    def __init__(self, factors=200, iterations=10, reg=0.01):
        super().__init__()
        self.factors = factors
        self.iterations = iterations
        self.reg = reg

    def compute_similarity(self, dataset):
        raise NotImplementedError

    def rate(self, dataset):
        model = als(factors=self.factors, iterations=self.iterations, regularization=self.reg)
        model.fit(dataset.T)
        ratings = np.dot(model.user_factors, model.item_factors.T)
        return sp.csr_matrix(ratings)







