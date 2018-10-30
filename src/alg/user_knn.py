"""
This file contains the UserKNN recommender which performs
a user-based collaborative filtering algorithm to determine the ranking
of each item for each user.
"""

import numpy as np
import scipy.sparse as sp
from timeit import default_timer as timer

from src.metrics import evaluate
from .recsys import RecSys
from .utils import cosine_similarity, predict


class UserKNN(RecSys):
    """
    User based recommender.

    Recommends base on the similarity between users
    """


    def __init__(self, alpha=0.5, asym=True, knn=np.inf, h=0, qfunc=None):
        """
        Constructor

        Parameters
        -----------
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
    
    
    def rate(self, dataset):

        print("computing similarity between users ...")
        start = timer()
        # Compute cosine similarity between users
        s = cosine_similarity(dataset.T, alpha=self.alpha, asym=self.asym, knn=self.knn, h=self.h, qfunc=self.qfunc, dtype=np.float32)
        print("elapsed time: {:.3f}s\n".format(timer() - start))

        print("computing ratings ...")
        start = timer()
        # Compute ratings
        ratings = (dataset.T * s).tocsr()
        print("elapsed time: {:.3f}s\n".format(timer() - start))
        del s

        return ratings.T