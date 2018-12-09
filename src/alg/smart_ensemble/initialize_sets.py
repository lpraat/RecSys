

# Initialize all the train sets and test sets for smart ensembling
import os

import pickle

from src.alg import ItemKNN, Hybrid, UserKNN
from src.alg.hybrid_similarity import HybridSimilarity
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

    ratings = model.rate(dataset.tocsr())
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


def initialize_sets(k=5):
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

        hybrid_graph = HybridSimilarity((P3Alpha(knn=1024, alpha=1), 0.15),
                                        (RP3Beta(knn=1024, alpha=1, beta=0.5), 0.95))
        write_preds(path=rel_path_preds, name="hybrid_graph", model=hybrid_graph, dataset=train_set)

        print("map")
        preds = build_preds(rel_path_preds, "hybrid_graph")
        print(evaluate(preds, test_set))

        hybrid_item_user = Hybrid((ItemKNN(("artist_set", 0.1, {}), ("album_set", 0.2, {})), 0.3),
                                  (UserKNN(knn=200), 0.2))
        write_preds(path=rel_path_preds, name="hybrid_item_user", model=hybrid_item_user, dataset=train_set)

        print("map")
        preds = build_preds(rel_path_preds, "hybrid_item_user")
        print(evaluate(preds, test_set))

        double_slim = HybridSimilarity((Slim(epochs=1, lr=0.1, lambda_i=0.01, all_dataset=False), 0.5),
                                       (Slim(epochs=1, lr=0.1, lambda_j=0.01, all_dataset=False), 0.5))
        write_preds(path=rel_path_preds, name="double_slim", model=double_slim, dataset=train_set)

        print("map")
        preds = build_preds(rel_path_preds, "double_slim")
        print(evaluate(preds, test_set))

        slim = Slim(epochs=3, lr=0.1, lambda_i=0.025, lambda_j=0.025, all_dataset=False)
        write_preds(path=rel_path_preds, name="slim", model=slim, dataset=train_set)

        print("map")
        preds = build_preds(rel_path_preds, "slim")
        print(evaluate(preds, test_set))

        h1 = HybridSimilarity((ItemKNN(("artist_set", 0.1, {}), ("album_set", 0.2, {})), 0.7),
                              (Slim(lambda_i=0.025, lambda_j=0.025, all_dataset=False, epochs=3, lr=0.1), 0.3))
        m1 = Hybrid((h1, 0.85), (UserKNN(knn=190), 0.15))
        write_preds(path=rel_path_preds, name="forzajuve", model=m1, dataset=train_set)

        print("map")
        preds = build_preds(rel_path_preds, "forzajuve")
        print(evaluate(preds, test_set))

