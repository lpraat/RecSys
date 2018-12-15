from src.alg.als import ALS
from src.alg.hybrid import Hybrid
from src.alg.hybrid_similarity import HybridSimilarity
from src.alg.item_knn import ItemKNN
from src.alg.light import Light
from src.alg.slim import Slim
from src.alg.user_knn import UserKNN
from src.data import Cache
import matplotlib.pyplot as plt

from src.metrics import ap_at_k, evaluate

cache = Cache()


def tracks_per_playlist():
    interactions = cache.fetch("interactions").tocsr()
    targets = cache.fetch("targets")

    res = {}
    for target in targets:
        playlist_len = interactions[target].count_nonzero()

        if playlist_len not in res:
            res[playlist_len] = playlist_len
        else:
            res[playlist_len] += 1
    return res


def targets_with_num(test=False):
    targets = cache.fetch("targets")

    if test:
        interactions = cache.fetch("train_set").tocsr()
    else:
        interactions = cache.fetch("train_set").tocsr()

    res = {}
    for target in targets:
        playlist_len = interactions[target].count_nonzero()
        res[target] = playlist_len

    return res


targets_num = targets_with_num()


def plot_tracks_per_playlist():
    res = tracks_per_playlist()
    playlist_len = [playlist_len for playlist_len, _ in res.items()]
    num = [num for _, num in res.items()]
    plt.bar(playlist_len, num)
    plt.show()


# Define a cluster split here
# automatically a cluster of [41, +infinite) is defined
clusters = [[0, 10], [11, 20], [21, 30], [31, 40]]


def determine_cluster(playlist):
    num = targets_num[playlist]

    for i, (lower, upper) in enumerate(clusters, 1):
        if lower <= num <= upper:
            return i
    return len(clusters) + 1


def tracks_per_cluster():
    targets = cache.fetch("targets")
    res = {}
    for target in targets:
        cluster = determine_cluster(target)
        if cluster not in res:
            res[cluster] = 1
        else:
            res[cluster] += 1

    return res


def plot_model_per_length(names_models):
    x_axis = list([i for i in range(1, len(clusters) + 2)])
    targets = cache.fetch("targets")
    train_set = cache.fetch("train_set")
    test_set = cache.fetch("test_set")

    models_preds = []

    for name, model in names_models:
        model_res = {}
        model_preds = model.run(dataset=train_set, targets=targets)
        print("MAP of " + str(name))
        print(evaluate(model_preds, test_set))

        for playlist, preds in model_preds.items():
            score = ap_at_k(preds, test_set[playlist])

            cluster = determine_cluster(playlist)

            if cluster in model_res:
                model_res[cluster] += [score]
            else:
                model_res[cluster] = [score]

        for k, v in model_res.items():
            model_res[k] = sum(v) / len(v)

        models_preds.append((name, model_res))

    for model_name, model_res in models_preds:
        values = [v for v in model_res.values()]
        print(values)
        plt.plot(x_axis, values, label=model_name)
        plt.plot()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Plot tracks per playlist
    # plot_tracks_per_playlist()

    def create(slim, light, als, w1, w2):
        h1 = HybridSimilarity((ItemKNN(("artist_set", 0.1, {}), ("album_set", 0.2, {})), 0.7),
                              (slim, 0.3))
        forzajuve = Hybrid((h1, 0.85), (UserKNN(knn=190), 0.15))

        return Hybrid((forzajuve, w1), (Hybrid((light, 0.5), (als,0.5)), w2))

    slim = Slim(lambda_i=0.025, lambda_j=0.025, all_dataset=False, epochs=3, lr=0.1)

    h1 = HybridSimilarity((ItemKNN(("artist_set", 0.1, {}), ("album_set", 0.2, {})), 0.7),
                          (slim, 0.3))
    forzajuve = Hybrid((h1, 0.85), (UserKNN(knn=190), 0.15))

    l = Light(no_components=300, epochs=30, loss='warp')
    a = ALS(factors=1024, iterations=2)

    la1 = create(slim, l, a, 0.9, 0.1)
    la2 = create(slim, l, a, 0.8, 0.2)
    la3 = create(slim, l, a, 0.7, 0.3)
    la4 = create(slim, l, a, 0.6, 0.4)
    la5 = create(slim, l, a, 0.5, 0.5)

    # Plot Models to evaluate on defined clusters
    print(tracks_per_cluster())
    plot_model_per_length([("forzajuve", forzajuve),
                           ("9-1", la1),
                           ("8-2", la2),
                           ("7-3", la3),
                           ("6-4", la4),
                           ("5-5", la5)])