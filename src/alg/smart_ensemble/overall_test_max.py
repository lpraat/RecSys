import os

import pickle

from src.alg.smart_ensemble.initialize_sets import build_preds
from src.metrics import evaluate

path = os.path.dirname(os.path.realpath(__file__)) + "/test"


def build_max_preds(model_names, rel_path):

    p = {}

    final_preds = {}

    for model_name in model_names:
        p[model_name] = build_preds(path, rel_path + "/" + model_name)

    with open(os.path.dirname(os.path.realpath(__file__)) + '/smart_max.csv', 'r') as f:
        lines = f.readlines()

        for line in lines:
            playlist, model = line[:-1].split(",")
            playlist = int(playlist)

            final_preds[playlist] = p[model][playlist]

    return final_preds


def test(k=5):

    for i in range(k):

        rel_path = path + "/" + str(i)
        rel_path_preds = rel_path + "/preds"
        rel_path_sets = rel_path + "/sets"

        with open(rel_path_sets + "/test.obj", "rb") as f:
            test_set = pickle.load(f)

        preds = build_max_preds(["hybrid_graph", "item_knn"], rel_path_preds)

        print(evaluate(preds, test_set))

test(5)


