from src.alg.als import ALS
from src.alg.hybrid import Hybrid
from src.alg.item_knn import ItemKNN
from src.data import Cache
import matplotlib.pyplot as plt

from src.metrics import ap_at_k

cache = Cache()
targets = cache.fetch("targets")
train_set = cache.fetch("train_set")
test_set = cache.fetch("test_set")
interactions = cache.fetch("interactions").tocsr()


def tracks_per_playlist():

    res = {}
    for target in targets:
        playlist_len = interactions[target].count_nonzero()

        if playlist_len not in res:
            res[playlist_len] = playlist_len
        else:
            res[playlist_len] += 1
    return res


def targets_with_num():
    targets = cache.fetch("targets")
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

    models_preds = []

    for name, model in names_models:
        model_res = {}
        model_preds = model.run(dataset=train_set, targets=targets)

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
        plt.plot(x_axis, values, label=model_name)
        plt.plot()
    plt.legend()
    plt.show()


# Plot tracks per playlist
# plot_tracks_per_playlist()

# Plot Models to evaluate on defined clusters
print(tracks_per_cluster())
plot_model_per_length([("item_knn", ItemKNN()), ("als", ALS(factors=200, iterations=50)),
                       ("hybrid", Hybrid((ItemKNN(), 0.9), (ALS(factors=200, iterations=50), 0.1)))])
