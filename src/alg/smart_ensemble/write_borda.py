import os

from src.alg.smart_ensemble.smart_ensemble import borda_count
from src.alg.smart_ensemble.utils import build_preds_from_file

max_file = os.path.dirname(os.path.realpath(__file__)) + '/smart_borda.csv'


def build_borda_preds(model_names, test=False):
    p = {}
    final_preds = {}

    for model_name in model_names:
        p[model_name] = build_preds_from_file(model_name, test)

    with open(max_file, 'r') as f:
        lines = f.readlines()

        for line in lines:
            playlist, model_w = line.split(",")
            playlist = int(playlist)

            models_preds = []
            weights = []

            for el in model_w.split("-")[:-1]:

                model_name, w = el.split("=")
                models_preds.append(p[model_name])
                weights.append(float(w))

            models_preds_and_weights = zip([preds[playlist] for preds in models_preds], weights)
            final_preds[playlist] = borda_count(models_preds_and_weights)

    return final_preds