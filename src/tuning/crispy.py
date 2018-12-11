from src.alg import ItemKNN, Hybrid, UserKNN
from src.alg.hybrid_similarity import HybridSimilarity
from src.alg.slim import Slim
from src.tuning.tuner import Callable, Hyperparameter, HyperparameterTuner

"""
# forzajuve
h1 = HybridSimilarity((ItemKNN(("artist_set", 0.1, {}), ("album_set", 0.2, {})), 0.7),
                      (Slim(lambda_i=0.025, lambda_j=0.025, all_dataset=False, epochs=3, lr=0.1), 0.3))
m1 = Hybrid((h1, 0.85), (UserKNN(knn=190), 0.15))


# planck
h1 = Hybrid((ItemKNN(("artist_set", 0.1, {}),("album_set", 0.2, {})),0.4),(UserKNN(knn=64), 0.2), normalize=False)
h2 = Hybrid((Slim(all_dataset=False, lr=0.1, lambda_i=0.01, lambda_j=0.01, epochs=3), 0.1), (h1, 0.9))

f = ((m1, 0.6), (h2, 0.4))
"""

# forzajuve
j_item_knn_callable = Callable(obj=ItemKNN, args=[("artist_set", 0.1, {}), ("album_set", 0.2, {})])
j_slim_callable = Callable(obj=Slim, kwargs={"lambda_i": 0.025, "lambda_j": 0.025, "all_dataset":False, "epochs":3, "lr":0.1})
j_h1_callable = Callable(obj=Hybrid, args=[(j_item_knn_callable, 0.7), (j_slim_callable, 0.3)])

j_user_callable = Callable(obj=UserKNN, kwargs={"knn": 200})
j_final = Callable(obj=Hybrid, args=[
    (j_h1_callable, 0.85), (j_user_callable, 0.15)
])

# planck
p_item_knn_callable = Callable(obj=ItemKNN, args=[("artist_set", 0.1, {}), ("album_set", 0.2, {})])
p_user_callable = Callable(obj=UserKNN, kwargs={"knn": 64})
p_h1_callable = Callable(obj=Hybrid, args=[
    (p_item_knn_callable, 0.4), (p_user_callable, 0.2)
])

p_slim_callable = Callable(obj=Slim, kwargs={"lambda_i": 0.01, "lambda_j": 0.01, "all_dataset": False, "epochs":3, "lr":0.1})
p_final = Callable(obj=Hybrid, args=[
    (p_h1_callable, 0.9), (p_slim_callable, 0.1)
])


w1 = Hyperparameter("w1-forzajuve", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
w2 = Hyperparameter("w2-planck", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
final_callable = Callable(obj=Hybrid, args=[
    (j_final, w1),
    (p_final, w2)
])

HyperparameterTuner(final_callable, cartesian_product=False).run()


