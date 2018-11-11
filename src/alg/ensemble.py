"""
This file contains the Ensemble recommender which combines different models.
"""

from timeit import default_timer as timer

import numpy as np
import scipy.sparse as sp
from timeit import default_timer as timer

from src.metrics import evaluate
from .recsys import RecSys


class Ensemble(RecSys):
    """
    Ensemble recommender.

    Recommends items to users by combining ratings from different
    models according to a defined distribution.
    """

    def __init__(self, *models):
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
        self.models = models
    

    def rate(self, dataset):

        # Compute combined ratings
        ratings = sp.csr_matrix(dataset.shape, dtype=np.float32)
        for model, w in self.models:
            model_ratings = model.rate(dataset)
            normalized_ratings = model_ratings.multiply(1/model_ratings.max())
            del model_ratings
            ratings += normalized_ratings * w
        
        return ratings
