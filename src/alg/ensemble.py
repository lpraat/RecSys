"""
This file contains the Ensemble recommender which combines different models.
"""

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

    def __init__(self, *models, method="combining"):
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
        self.method = method

    def run(self, dataset=None, targets=None, k=10):

        # Run each model
        ensemble_preds = []
        for model in self.models:
            ensemble_preds.append(model.run(dataset=dataset, targets=targets, k=k))

        if self.method == "combining":
            n = len(self.models)
            # Sanity check
            assert n <= k, "too many models for 'combining' method (max. {})".format(k)

            print("combining results ...")
            start = timer()
            # Combining list predictions
            preds = []
            for t in range(len(ensemble_preds[0])):
                pred = []
                i = 0
                while len(pred) < k:
                    # Get prediction from i-th model
                    item = ensemble_preds[i % n][t][1][i // n]
                    if item not in pred:
                        pred.append(item)
                    i += 1

                preds.append((ensemble_preds[0][t][0], pred))
            print("elapsed time: {:.3f}\n".format(timer() - start))
            
            return preds
        
        else:
            raise NotImplementedError("method '{}' not implemented".format(self.method))
