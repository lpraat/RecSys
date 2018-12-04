import numpy as np

from src.alg import Hybrid, ItemKNN, UserKNN
from src.alg.p3alpha import P3Alpha
from src.alg.rp3beta import RP3Beta
from src.alg.slim import Slim
from src.alg.svd import SVD
from src.tuning.tuner import Callable, HyperparameterTuner, Hyperparameter


""" Tuned model
m1 = SVD(factors=200, knn=500)
m2 = ItemKNN(("album_set", 0.2, {}), ("artist_set", 0.1, {}))
m3 = Slim(lambda_j=0.025, lr=0.01, epochs=1, all_dataset=False)
m4 = Slim(lambda_i=0.025, lr=0.01, epochs=1, all_dataset=False)
m5 = UserKNN(knn=200)

Hybrid((m1, w1), (m2, w2)..., normalize=True/False)
"""

svd_callable = Callable(SVD, [], {"factors": 200, "knn": 512})
svd_weight = Hyperparameter("svd_weight", np.arange(0.1, 1, 0.05))

item_knn_callable = Callable(ItemKNN, [("album_set", 0.2, {}), ("artist_set", 0.1, {})])
item_knn_weight = Hyperparameter("item_knn_weight", np.arange(0.1, 1, 0.05))

slim1_callable = Callable(Slim, [], {"lambda_i": 0.025, "lr": 0.01, "epochs": 1, "all_dataset": False})
slim1_weight = Hyperparameter("slim1_weight", np.arange(0.1, 1, 0.05))

slim2_callable = Callable(Slim, [], {"lambda_j": 0.025, "lr": 0.01, "epochs": 1, "all_dataset": False})
slim2_weight = slim1_weight

user_knn_knn = Hyperparameter("item_knn", [64, 128, 256, 512])
user_knn_callable = Callable(UserKNN, [], {"knn": user_knn_knn})
user_knn_weight = Hyperparameter("user_knn_weight", np.arange(0.1, 1, 0.05))

palpha_knn = Hyperparameter("palpha_knn", [128, 256, 512, 1024])
palpha_callable = Callable(P3Alpha, [], {"knn": palpha_knn, "alpha": 1})
palpha_weight = Hyperparameter("palpha_weight", np.arange(0.1, 1, 0.05))

rbeta_knn = Hyperparameter("rbeta_knn", [128, 256, 512, 1024])
rbeta_callable = Callable(RP3Beta, [], {"knn": rbeta_knn, "alpha": 1, "beta": 0.3})
rbeta_weight = Hyperparameter("rbeta_weight", np.arange(0.1, 1, 0.05))

hybrid = Callable(Hybrid, [
    (svd_callable, svd_weight),
    (item_knn_callable, item_knn_weight),
    (slim1_callable, slim1_weight),
    (slim2_callable, slim2_weight),
    (user_knn_callable, user_knn_weight),
    (palpha_callable, palpha_weight),
    (rbeta_callable, rbeta_weight)
])

t = HyperparameterTuner(hybrid, cartesian_product=False)
t.run()
