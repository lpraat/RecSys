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

        if self.method == "combining":
            n = len(self.models)

            # Run each model
            ensemble_preds = []
            for model in self.models:
                ensemble_preds.append(model.run(dataset=dataset, targets=targets, k=k))

            print("combining results ...")
            start = timer()
            # Combining list predictions
            preds = []
            for ti in range(len(ensemble_preds[0])):
                pred = []
                i = 0
                while len(pred) < k:
                    # Get prediction from i-th model
                    item = ensemble_preds[i % n][ti][1][i // n]
                    if item not in pred:
                        pred.append(item)
                    i += 1

                preds.append((ensemble_preds[0][ti][0], pred))
            print("elapsed time: {:.3f}\n".format(timer() - start))

            return preds

        elif self.method == "roundrobin":
            n = len(self.models)

            # Run each model
            ensemble_preds = []
            for model, _ in self.models:
                ensemble_preds.append(model.run(dataset=dataset, targets=targets, k=k))

            # Normalize probabilities
            cdf = np.array([m[1] for m in self.models])
            cdf /= cdf.sum()

            print("combining results with probabilities {} ...".format(cdf))
            start = timer()
            # Combine list predictions
            preds = []
            for ti in range(len(ensemble_preds[0])):
                # Generate pick strategy with probabilities cdf
                choices = np.random.choice(n, size=k, p=cdf)

                pred = []
                offset = [0 for _ in range(n)]
                for mi in choices:
                    # Ensure no duplicates
                    item = ensemble_preds[mi][ti][1][offset[mi]]; offset[mi] += 1
                    while item in pred:
                        item = ensemble_preds[mi][ti][1][offset[mi]]; offset[mi] += 1

                    # Append to predictions
                    pred.append(item)

                preds.append((ensemble_preds[0][ti][0], pred))
            print("elapsed time: {:.3f}\n".format(timer() - start))

            return preds

        else:
            raise NotImplementedError("method '{}' not implemented".format(self.method))
