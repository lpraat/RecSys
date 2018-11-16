"""
This file contains the Ensemble recommender which combines different models.
"""

import numpy as np
import scipy.sparse as sp
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

        '''
        TODO
        I'm still not sure about the code below.
        By using Slim + CBF & CB it works on 16 gb ram.
        But I'm pretty sure that if we 
        '''
        while self.models:

            model, w = self.models.pop()
            model_ratings = model.rate(dataset)

            if self.normalize:
                model_ratings = model_ratings.multiply(1 / model_ratings.max())

            model_ratings = model_ratings * w
            ratings += model_ratings
            del model_ratings

        return ratings
