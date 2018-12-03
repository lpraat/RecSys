from src.alg import ItemKNN, UserKNN, Hybrid
from src.alg.hybrid_similarity import HybridSimilarity
from src.alg.slim import Slim
from src.alg.svd import SVD
from src.tuning.tuner import Callable, Hyperparameter, HyperparameterTuner


""" Hyperparameter tuning for 
h1 = HybridSimilarity((SVD(factors=200, knn=1000), 0.1),
                      (Slim(all_dataset=False, lr=0.1, lambda_i=0.01), 0.2),
                      (Slim(all_dataset=False, lr=0.1, lambda_j=0.01), 0.3),
                      (ItemKNN(("artist_set", 0.1, {}), ("album_set", 0.2, {})), 0.4), normalize=True)

h2 = UserKNN(knn=200).evaluate()
h3 = Hybrid((h1, 0.2), (h2, 0.3), normalize=False)

"""
svd_callable = Callable(SVD, [], {"factors": 200, "knn": 1000})
svd_weight = Hyperparameter("svd_weight", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

item_knn_callable = Callable(ItemKNN, [("album_set", 0.2, {}), ("artist_set", 0.1, {})])
item_knn_weight = Hyperparameter("item_knn_weight", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

slim_i_callable = Callable(Slim, [], kwargs={"all_dataset": False, "epochs": 1, "lr": 0.1, "lambda_i": 0.01})
slim_i_weight = Hyperparameter("slim_i_weight", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

slim_j_callable = Callable(Slim, [], kwargs={"all_dataset": False, "epochs": 1, "lr": 0.1, "lambda_j": 0.01})
slim_j_weight = slim_i_weight  # assign the same weight in each run to both Slim

hybrid_similarity_normalize = Hyperparameter('hybrid_similarity_normalize', [True, False])
hybrid_similarity_callable = Callable(HybridSimilarity, [
    (svd_callable, svd_weight),
    (item_knn_callable, item_knn_weight),
    (slim_i_callable, slim_i_weight),
    (slim_j_callable, slim_j_weight)
], kwargs={"normalize": hybrid_similarity_normalize})
hybrid_similarity_weight = Hyperparameter("hybrid_similarity_weight", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

user_knn_callable = Callable(UserKNN, [], {"knn": 200})
user_knn_weight = Hyperparameter("user_knn_weight", [0.01, 0.02, 0.05, 0.1, 0.15, 0.2])

final_hybrid_normalize = Hyperparameter('final_hybrid_normalize', [True, False])
final_hybrid = Callable(
    obj=Hybrid,
    args=[(hybrid_similarity_callable, hybrid_similarity_weight), (user_knn_callable, user_knn_weight)],
    kwargs={
        "normalize": final_hybrid_normalize
    }
)

tune = HyperparameterTuner(final_hybrid, cartesian_product=False)
tune.run()