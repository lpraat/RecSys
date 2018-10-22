"""
Combine (the predictions of) two or more models
"""

import random
import bisect
import numpy as np
from timeit import default_timer as timer

from .recsys import RecSys
from src.metrics import evaluate


class Ensamble(RecSys):
    """
    An ensamble model simply combines the results of tow or more algorithms
    picking from the predictions with different probabilities
    """


    def __init__(self, dataset = "train_set", models = []):
        """ Class constructor """

        # Super constructor
        super().__init__(dataset)

        # Initial values
        self.dataset    = dataset
        self.models     = models
    

    def run(self, targets, k = 10):
        
        # Compute predictions for all models
        preds = []
        for model, _ in self.models:

            # Sanity check
            assert isinstance(model, RecSys)

            # Run model
            model.dataset = self.dataset
            pred = model.run(targets, k)
            preds.append(pred)
        
        # Compute normalized probabilities
        cdf = np.array([model[1] for model in self.models])
        cdf /= sum(cdf)

        print(len(cdf))
        
        print("computing final predictions with probabilities {} ...".format(cdf))
        start = timer()
        # Cherry pick with probabilities p from predictions
        for ti in range(len(targets)):
            # Generate k pick indices
            indices = np.random.choice(len(cdf), k, p = cdf)
            for pi, m in enumerate(indices):
                if m != 0:
                    preds[0][ti][1][pi] = preds[m][ti][1][pi]

        preds = preds[0]
        print("elapsed time: {:.3}s\n".format(timer() - start))

        # @debug
        # Estimate model
        score = evaluate(preds, self.cache.fetch("test_set"))
        print("MAP@{}: {:.5}\n".format(k, score))

        return preds