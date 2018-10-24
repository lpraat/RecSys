"""
Combine (the predictions of) two or more models
"""

import numpy as np
import scipy.sparse as sp
from timeit import default_timer as timer

from .recsys import RecSys
from src.metrics import evaluate


class Ensemble(RecSys):
    """ An ensemble model simply combines the ratings of two or more models """


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
            ratings += model.rate(dataset) * w
        
        return ratings