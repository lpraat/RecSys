import os

from src.alg.smart_ensemble.utils import build_preds_from_file

max_file = os.path.dirname(os.path.realpath(__file__)) + '/smart_max.csv'


def build_max_preds(model_names, test=False):
    p = {}

    final_preds = {}

    for model_name in model_names:
        p[model_name] = build_preds_from_file(model_name, test)

    with open(max_file, 'r') as f:
        lines = f.readlines()

        for line in lines:
            playlist, model = line.split(",")
            playlist = int(playlist)

            final_preds[playlist] = p[model][playlist]

    return final_preds








