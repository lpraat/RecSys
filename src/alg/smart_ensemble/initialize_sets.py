

# Initialize all the train sets and test sets for smart ensembling
import os

import pickle

from src.alg import ItemKNN, Hybrid, UserKNN
from src.alg.als import ALS
from src.alg.hybrid_similarity import HybridSimilarity
from src.alg.light import Light
from src.alg.p3alpha import P3Alpha
from src.alg.rp3beta import RP3Beta
from src.alg.slim import Slim
from src.alg.utils import predict
from src.data import Cache, build_train_set_uniform
from src.metrics import evaluate

cache = Cache()

path = os.path.dirname(os.path.realpath(__file__)) + "/validate"


def evaluate_preds(preds, test_set):
    return evaluate(preds, test_set)


def write_preds(path, name, model, dataset, targets=cache.fetch("targets")):

    ratings = model.rate(dataset.tocsr(), targets)
    preds = predict(ratings, targets=targets, k=10, mask=dataset, invert_mask=True)
    del ratings
    # Create directory if necessary
    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, name + ".csv"), 'w') as f:
        f.write("playlist_id,track_ids\n")
        for playlist, values in preds.items():
            f.write(str(playlist) + ",")
            f.write(" ".join([str(el) for el in values]))
            f.write("\n")


def build_preds(path, filename):
    preds = {}

    with open(os.path.join(path, filename + ".csv"), 'r') as f:

        lines = f.readlines()[1:]

        for line in lines:
            playlist, tracks = line.split(",")
            playlist = int(playlist)
            tracks = [int(x) for x in tracks.split(" ")]
            preds[playlist] = tracks

    return preds


def initialize_sets(k=10):
    for i in range(k):

        print(i)

        rel_path = path + "/" + str(i)
        rel_path_sets = rel_path + "/sets"
        rel_path_preds = rel_path + "/preds"

        os.makedirs(rel_path, exist_ok=True)
        os.makedirs(rel_path_sets, exist_ok=True)
        os.makedirs(rel_path_preds, exist_ok=True)

        train_set, test_set = build_train_set_uniform(cache.fetch("interactions"), cache.fetch("targets"), 0.2)

        with open(rel_path_sets + "/train.obj", "wb") as f:
            pickle.dump(train_set, f)

        with open(rel_path_sets + "/test.obj", "wb") as f:
            pickle.dump(test_set, f)

        def create(w1, w2):
            slim = Slim(lambda_i=0.025, lambda_j=0.025, all_dataset=False, epochs=3, lr=0.1)
            light = Light(no_components=300, epochs=30, loss='warp')
            als = ALS(factors=1024, iterations=2)

            h1 = HybridSimilarity((ItemKNN(("artist_set", 0.1, {}), ("album_set", 0.2, {})), 0.7),
                                  (slim, 0.3))
            forzajuve = Hybrid((h1, 0.85), (UserKNN(knn=190), 0.15))

            return Hybrid((forzajuve, w1), (Hybrid((light, 0.5), (als, 0.5)), w2))

        m1 = create(0.8, 0.2)
        write_preds(path=rel_path_preds, name="8-2", model=m1, dataset=train_set)
        print("map")
        preds = build_preds(rel_path_preds, "8-2")
        print(evaluate(preds, test_set))

        m2 = create(0.7, 0.3)
        write_preds(path=rel_path_preds, name="7-3", model=m2, dataset=train_set)
        print("map")
        preds = build_preds(rel_path_preds, "7-3")
        print(evaluate(preds, test_set))

        m3 = create(0.6, 0.4)
        write_preds(path=rel_path_preds, name="6-4", model=m3, dataset=train_set)
        print("map")
        preds = build_preds(rel_path_preds, "6-4")
        print(evaluate(preds, test_set))

        m4 = create(0.4, 0.6)
        write_preds(path=rel_path_preds, name="4-6", model=m4, dataset=train_set)
        print("map")
        preds = build_preds(rel_path_preds, "4-6")
        print(evaluate(preds, test_set))

        m5 = create(0.3, 0.7)
        write_preds(path=rel_path_preds, name="3-7", model=m5, dataset=train_set)
        print("map")
        preds = build_preds(rel_path_preds, "3-7")
        print(evaluate(preds, test_set))

        m6 = create(0.2, 0.8)
        write_preds(path=rel_path_preds, name="2-8", model=m6, dataset=train_set)
        print("map")
        preds = build_preds(rel_path_preds, "2-8")
        print(evaluate(preds, test_set))



