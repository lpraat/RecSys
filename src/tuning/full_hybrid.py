import numpy as np

from src.alg import Hybrid, ItemKNN, UserKNN
from src.alg.als import ALS
from src.alg.light import Light
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

item_knn_callable = Callable(ItemKNN, [("album_set", 0.2, {}), ("artist_set", 0.1, {})])
item_knn_weight = Hyperparameter("item_knn_weight", np.arange(0.1, 1, 0.05))

slim_callable = Callable(Slim, [], {"lambda_i": 0.01, "lambda_j": 0.01, "lr": 0.01, "epochs": 3, "all_dataset": False})
slim_weight = Hyperparameter("slim_weight", np.arange(0.1, 1, 0.05))

user_knn_callable = Callable(UserKNN, [], {"knn": 200})
user_knn_weight = Hyperparameter("user_knn_weight", np.arange(0.1, 1, 0.05))

als_callable = Callable(ALS, [], {"num_factors": 200, "iterations": 10})
als_weight = Hyperparameter("als_weight", np.arange(0.1, 1, 0.05))

rbeta_callable = Callable(RP3Beta, [], {"knn": 128, "alpha": 1, "beta": 0.3})
rbeta_weight = Hyperparameter("rbeta_weight", np.arange(0.1, 1, 0.05))

hybrid = Callable(Hybrid, [
    (item_knn_callable, item_knn_weight),
    (slim_callable, slim_weight),
    (user_knn_callable, user_knn_weight),
    (als_callable, als_weight),
    (rbeta_callable, rbeta_weight)
])

t = HyperparameterTuner(hybrid, cartesian_product=False)
t.run()
