"""
This file contains the Ensemble recommender which ensembles different models.
"""
from timeit import default_timer as timer

import numpy as np

from src.metrics import evaluate
from .recsys import RecSys


class Ensemble(RecSys):
    """Ensemble recommender.

    Recommends items to users by sampling recommendations from different
    models according to a probability distribution defined on them.


    Attributes
    ----------
    dataset : str
        Name of the dataset from cache to be used to generate recommendations.
    models : list
        The ensembled models.

    """

    def __init__(self, dataset="interactions", models=[]):
        """ Class constructor """

        # Super constructor
        super().__init__(dataset)

        # Initial values
        self.dataset = dataset
        self.models = models

    def run(self, targets=None, k=10):
        """ Get k predictions for each target user """

        # Sanity check
        assert len(self.models) > 0, "no model specified"

        # Compute predictions for all models
        preds = []
        for model, _ in self.models:
            # Sanity check
            assert isinstance(model, RecSys), "model {} is not an instance of RecSys".format(type(model))

            # Run model
            model.dataset = self.dataset
            preds.append(model.run(targets=targets, k=k))

        # Compute normalized probabilities
        cdf = np.array([model[1] for model in self.models])
        cdf /= sum(cdf)

        print("computing final predictions with probabilities {} ...".format(cdf))
        start = timer()
        preds_final = []
        # Cherry pick with probabilities p from predictions
        for ti in range(len(preds[0])):
            # Generate k pick indices
            choices = np.random.choice(len(cdf), k, p=cdf)

            pred_ti = []
            for m in choices:
                # Get a non-duplicate prediction
                pred_pi = preds[m][ti][1].pop(0)
                while pred_pi in pred_ti:
                    pred_pi = preds[m][ti][1].pop(0)

                pred_ti.append(pred_pi)

            # Append to final prediction
            preds_final.append((preds[0][ti][0], pred_ti))

        print("elapsed time: {:.3f}s\n".format(timer() - start))

        # Return predictions
        return preds_final

    def evaluate(self, train_set="train_set", test_set="test_set", k=10):
        """ Evaluate model performance using MAP@k """

        # Sanity check
        assert len(self.models) > 0, "no model specified"

        # Compute predictions for all models
        preds = []
        for model, _ in self.models:
            # Sanity check
            assert isinstance(model, RecSys), "model {} is not an instance of RecSys".format(type(model))

            # Run model
            model.dataset = train_set
            preds.append(model.run(k=k))

        # Compute normalized probabilities
        cdf = np.array([model[1] for model in self.models])
        cdf /= sum(cdf)

        print("computing final predictions with probabilities {} ...".format(cdf))
        start = timer()
        preds_final = []
        # Cherry pick with probabilities p from predictions
        for ti in range(len(preds[0])):
            # Generate k pick indices
            choices = np.random.choice(len(cdf), k, p=cdf)

            pred_ti = []
            for m in choices:
                # Get a non-duplicate prediction
                pred_pi = preds[m][ti][1].pop(0)
                while pred_pi in pred_ti:
                    pred_pi = preds[m][ti][1].pop(0)

                pred_ti.append(pred_pi)

            # Append to final prediction
            preds_final.append((ti, pred_ti))

        print("elapsed time: {:.3f}s\n".format(timer() - start))

        # Evaluate model
        score = evaluate(preds_final, self.cache.fetch(test_set))
        print("MAP@{}: {:.5}\n".format(k, score))

        return score
