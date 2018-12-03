import numpy as np

from src.alg import ItemKNN, UserKNN, Hybrid
from src.alg.hybrid_similarity import HybridSimilarity
from src.alg.p3alpha import P3Alpha
from src.alg.rp3beta import RP3Beta
from src.alg.slim import Slim
from src.alg.svd import SVD
from src.tuning.tuner import Callable, Hyperparameter, HyperparameterTuner


""" Hyperparameter tuning for 
h_graph = 
h1 = HybridSimilarity((SVD(factors=200, knn=1000), 0.1),
                      (Slim(all_dataset=False, lr=0.1, lambda_i=0.01), 0.2),
                      (Slim(all_dataset=False, lr=0.1, lambda_j=0.01), 0.3),
                      (ItemKNN(("artist_set", 0.1, {}), ("album_set", 0.2, {})), 0.4),
                      (P3Alpha(knn=1000, alpha=1), 0.01), 
                      (RP3Beta(knn=1000, alpha=1, beta=0.3), 0.99))

h1.evaluate()

"""

weights = np.arange(0.1, 1, 0.05)

svd_knn = Hyperparameter("svd_knn", [16, 32, 64, 128, 256, 512])
svd_callable = Callable(SVD, [], {"factors": 200, "knn": svd_knn})
svd_weight = Hyperparameter("svd_weight", weights)

item_knn_knn = Hyperparameter("item_knn_knn", [128, 256, 512, 1024])
item_knn_callable = Callable(ItemKNN, [("album_set", 0.2, {}), ("artist_set", 0.1, {})], kwargs={"knn": item_knn_knn})
item_knn_weight = Hyperparameter("item_knn_weight", weights)

slim_knn = Hyperparameter("slim_knn", [128, 256, 512, 1024])
slim_i_callable = Callable(Slim, [], kwargs={"all_dataset": False, "epochs": 1, "lr": 0.1, "lambda_i": 0.01, "knn": slim_knn})
slim_i_weight = Hyperparameter("slim_i_weight", weights)

slim_j_callable = Callable(Slim, [], kwargs={"all_dataset": False, "epochs": 1, "lr": 0.1, "lambda_j": 0.01, "knn": slim_knn})
slim_j_weight = slim_i_weight  # assign the same weight in each run to both Slim

palpha_knn = Hyperparameter("palpha_knn", [128, 256, 512, 1024])
palpha_callable = Callable(P3Alpha, [], {"knn": palpha_knn, "alpha":1})
palpha_weight = Hyperparameter("palpha_weight", weights)

rbeta_knn = Hyperparameter("rbeta_knn", [128, 256, 512, 1024])
rbeta_callable = Callable(RP3Beta, [], {"knn": rbeta_knn, "alpha": 0.1, "beta": 0.3})
rbeta_weight = Hyperparameter("rbeta_weight", weights)

hybrid_similarity_callable = Callable(HybridSimilarity, [
    (svd_callable, svd_weight),
    (item_knn_callable, item_knn_weight),
    (slim_i_callable, slim_i_weight),
    (slim_j_callable, slim_j_weight),
    (palpha_callable, palpha_weight),
    (rbeta_callable, rbeta_weight)
])

tune = HyperparameterTuner(hybrid_similarity_callable, cartesian_product=False)
tune.run()