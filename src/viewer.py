from src.alg.hybrid import Hybrid
from src.alg.hybrid_similarity import HybridSimilarity
from src.alg.item_knn import ItemKNN
from src.alg.slim import Slim
from src.alg.user_knn import UserKNN
from src.data import Cache
import matplotlib.pyplot as plt

from src.metrics import ap_at_k

cache = Cache()


def tracks_per_playlist():
    targets = cache.fetch("targets")
    interactions = cache.fetch("interactions").tocsr()

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


def plot_tracks_per_playlist():
    res = tracks_per_playlist()
    playlist_len = [playlist_len for playlist_len, _ in res.items()]
    num = [num for _, num in res.items()]
    plt.bar(playlist_len, num)
    plt.show()


def plot_model_per_length(names_models):
    targets = cache.fetch("targets")
    train_set = cache.fetch("train_set")
    targets_num = targets_with_num()
    test_set = cache.fetch("test_set")
    x_axis = list(set([v for v in targets_num.values()]))

    models_preds = []

    for name, model in names_models:
        model_res = {}
        model_preds = model.run(dataset=train_set, targets=targets)

        for playlist, preds in model_preds.items():
            score = ap_at_k(preds, test_set[playlist])

            if targets_num[playlist] in model_res:
                model_res[targets_num[playlist]] += [score]
            else:
                model_res[targets_num[playlist]] = [score]

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

# Models to evaluate
# forzajuve.csv in kaggle
h1 = HybridSimilarity((ItemKNN(("artist_set", 0.1, {}), ("album_set", 0.2, {})), 0.7),
                      (Slim(lambda_i=0.025, lambda_j=0.025, all_dataset=False, epochs=3, lr=0.1), 0.3))
m1 = Hybrid((h1, 0.85), (UserKNN(knn=190), 0.15))

# user_item hybrid
ui = Hybrid((ItemKNN(("artist_set", 0.1, {}),("album_set", 0.2, {})),0.4),(UserKNN(knn=64), 0.2), normalize=False)

plot_model_per_length([("forzajuve", m1),
                       ("user_item_hybrid", ui)])

