import os

from src.alg.recsys import RecSys
from src.alg.smart_ensemble.utils import build_preds_from_file
from src.data import Cache
from src.metrics import evaluate


class SmartEnsemble():

    def __init__(self, models_from_files_and_weights, test=False):
        super().__init__()

        # Global cache
        global cache

        # Create cache if necessary
        # This ensures that only one global cache exists
        # Don't reuse memory!
        try:
            self.cache = cache
        except NameError:
            cache = Cache()
            self.cache = cache

        self.models_preds = []
        self.weights = []

        for model_file, w in models_from_files_and_weights:
            self.models_preds.append(build_preds_from_file(model_file, test=test))
            self.weights.append(w)

    def merge_preds(self):
        targets = self.cache.fetch("targets")

        final_preds = {}
        for playlist_id in targets:
            models_preds_and_weights = zip([preds[playlist_id] for preds in self.models_preds], self.weights)
            final_preds[playlist_id] = borda_count(models_preds_and_weights)

        return final_preds

    def evaluate(self):
        final_preds = self.merge_preds()
        score = evaluate(final_preds, self.cache.fetch("test_set"))
        print("MAP@{}: {:.5f}\n".format(10, score))
        return score


def borda_count(models_preds_and_weights, n=10):
    scores = {}
    for model_preds, w in models_preds_and_weights:
        for i in range(len(model_preds)):

            if model_preds[i] in scores:
                scores[model_preds[i]] += (n - i) * w
            else:
                scores[model_preds[i]] = (n - i) * w

    results = reversed(sorted([(track, score) for track, score in scores.items()], key=lambda x: x[1]))
    return [track for track, _ in results]















