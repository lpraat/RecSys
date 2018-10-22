"""
Combine (the predictions of) two or more models
"""

import random
import bisect
import numpy as np
from timeit import default_timer as timer

from .recsys import RecSys
from src.metrics import evaluate


class Ensemble(RecSys):
    """
    An ensemble model simply combines the results of tow or more algorithms
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
        
        print("computing final predictions with probabilities {} ...".format(cdf))
        start = timer()
        preds_final = []
        # Cherry pick with probabilities p from predictions
        for ti in range(len(targets)):
            # Generate k pick indices
            choices = np.random.choice(len(cdf), k, p = cdf)
            
            pred_ti = []
            for m in choices:
                pred_pi = preds[m][ti][1].pop(0)
                while pred_pi in pred_ti:
                    pred_pi = preds[m][ti][1].pop(0)
                
                pred_ti.append(pred_pi)
            
            preds_final.append((targets[ti], pred_ti))

        print("elapsed time: {:.3}s\n".format(timer() - start))

        # Return predictions
        return preds_final

    
    def evaluate(self, train_set = None):


        # @todo
        # Evaluate model
        score = evaluate(preds, self.cache.fetch("test_set"))
        print("MAP@{}: {:.5}\n".format(k, score))