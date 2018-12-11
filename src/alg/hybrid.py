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
    models according to a defined distribution.
    """

    def __init__(self, *models, normalize=True):
        """
        Constructor

        Parameters
        -----------
        *models : RecSys
            List of recommender system to combine when computing final ratings
        """

        # Super constructor
        super().__init__()

        # Initial values
        self.models = list(models)
        self.normalize = normalize

    def rate(self, dataset):

        if not self.models:
            raise RuntimeError("You already called rate")

        # Compute combined ratings
        ratings = sp.csr_matrix(dataset.shape, dtype=np.float32)

        while self.models:

            model, w = self.models.pop()
            model_ratings = model.rate(dataset)

            if self.normalize:

                for i in range(ratings.shape[0]):
                    start_data = model_ratings.indptr[i]
                    end_data = model_ratings.indptr[i+1]

                    row_data = model_ratings.data[start_data:end_data]
                    row_max = np.max(row_data)
                    row_data /= row_max
                    row_data *= w

            del model_ratings

        return ratings
