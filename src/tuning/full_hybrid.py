from src.alg import Hybrid, ItemKNN, UserKNN
from src.alg.slim import Slim
from src.alg.svd import SVD
from src.tuning.tuner import Callable, HyperparameterTuner, Hyperparameter

m1 = SVD(factors=200, knn=500)
m2 = ItemKNN(("album_set", 0.2, {}), ("artist_set", 0.1, {}))
m3 = Slim(lambda_j=0.025, lr=0.01, epochs=1, all_dataset=False)
m4 = Slim(lambda_i=0.025, lr=0.01, epochs=1, all_dataset=False)
m5 = UserKNN(knn=200)

Hybrid()


svd_callable = Callable(SVD, [], {"factors": 200, "knn": 500})
svd_weight = Hyperparameter("svd_weight", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

item_knn_callable = Callable(ItemKNN, [("album_set", 0.2, {}), ("artist_set", 0.1, {})], {})
item_knn_weight = Hyperparameter("item_knn_weight", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

slim1_callable = Callable(Slim, [], {"lambda_i": 0.025, "lr": 0.01, "epochs": 1, "all_dataset": False})
slim1_weight = Hyperparameter("slim1_weight", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


slim2_callable = Callable(Slim, [], {"lambda_j": 0.025, "lr": 0.01, "epochs": 1, "all_dataset": False})
slim2_weight = Hyperparameter("slim2_weight", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

hybrid_normalize = Hyperparameter("hybrid_normalize", [True, False])
hybrid = Callable(Hybrid, [
    (svd_callable, svd_weight),
    (item_knn_callable, item_knn_weight),
    (slim1_callable, slim1_weight),
    (slim2_callable, slim2_weight)
], kwargs={"normalize": hybrid_normalize})

t = HyperparameterTuner(hybrid, cartesian_product=False)
