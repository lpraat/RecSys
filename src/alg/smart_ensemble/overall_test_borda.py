import os

import pickle

from src.alg.smart_ensemble.intialize_sets import build_preds
from src.alg.smart_ensemble.smart_ensemble import borda_count
from src.metrics import evaluate

path = os.path.dirname(os.path.realpath(__file__)) + "/test"


def build_borda_preds(model_names, rel_path):
    p = {}
    final_preds = {}

    for model_name in model_names:
        p[model_name] = build_preds(model_name, rel_path + "/" + model_name)

    with open(os.path.dirname(os.path.realpath(__file__)) + '/smart_borda.csv', 'r') as f:
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


def test(k=5):

    for i in range(k):

        rel_path = path + "/" + str(i)
        rel_path_preds = rel_path + "/preds"
        rel_path_sets = rel_path + "/sets"

        with open(rel_path_sets + "/test.obj", "rb") as f:
            test_set = pickle.load(f)

        preds = build_borda_preds(["hybrid_graph", "item_knn"], rel_path_preds)

        print(evaluate(preds, test_set))


test(5)

