"""
This file contains the Ensemble recommender which combines different models.
"""

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize

from .recsys import RecSys


class Hybrid(RecSys):
    """
    Ensemble recommender.

    Recommends items to users by combining ratings from different
    models.
    """

    def __init__(self, *models, normalize=True):
        """
        Constructor

        Parameters
        -----------
        *models : RecSys
            List of recommender systems to combine when computing final ratings
        """

        # Super constructor
        super().__init__()

        # Initial values
        self.models = list(models)
        self.normalize = normalize

    def rate(self, dataset, targets):
        if not self.models:
            raise RuntimeError("You already called rate")

        # Compute combined ratings
        ratings = sp.csr_matrix((len(targets), dataset.shape[1]), dtype=np.float32)

        while self.models:

            model, w = self.models.pop()
            model_ratings = model.rate(dataset, targets).tocsr()

            if self.normalize:
                model_ratings = normalize(model_ratings, norm='l2', axis=1)

            model_ratings = model_ratings * w
            ratings += model_ratings
            del model_ratings

        return ratings
