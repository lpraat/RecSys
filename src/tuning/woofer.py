from src.alg import ItemKNN, UserKNN, Hybrid
from src.alg.als import ALS
from src.alg.hybrid_similarity import HybridSimilarity
from src.alg.light import Light
from src.alg.slim import Slim
from src.tuning.tuner import Callable, Hyperparameter, HyperparameterTuner

slim_callable = Callable(obj=Slim, kwargs={"lambda_i":0.025, "lambda_j":0.025, "all_dataset":False, "epochs":3, "lr":0.1})
item_callable = Callable(obj=ItemKNN, args=[("artist_set", 0.1, {}), ("album_set", 0.2, {})])

slim_weight = Hyperparameter("slim_weight", [0.3, 0.2, 0.25])
item_weight = Hyperparameter("item_weight", [0.7, 0.8, 0.75])
h1_callable = Callable(obj=HybridSimilarity, args=[
    (slim_callable, slim_weight),
    (item_callable, item_weight)
])

user_knn = Hyperparameter("user_knn", [64, 200])
user_callable = Callable(obj=UserKNN, kwargs={"knn": user_knn})

h1_weight = Hyperparameter("h1_weight", [0.85, 0.9])
user_weight = Hyperparameter("user_weight", [0.15, 0.1])

forzajuve_callable = Callable(obj=Hybrid, args=[
    (h1_callable, h1_weight),
    (user_callable, user_weight)
])

light_knn = Hyperparameter("light_knn", [256, 512, 1024])
light_callable = Callable(obj=Light, kwargs={"no_components":300, "epochs":30, "loss":"warp", "knn": light_knn})

als_knn = Hyperparameter("als_knn", [256, 512, 1024])
als_callable = Callable(obj=ALS, kwargs={"factors":1024, "iterations":2, "knn": als_knn})

light_weight = Hyperparameter("light_weight", [0.7, 0.6, 0.4, 0.5])
als_weight = Hyperparameter("als_weight", [0.3, 0.4, 0.6, 0.5])

la_callable = Callable(obj=Hybrid, args=[
    (light_callable, light_weight),
    (als_callable, als_weight)
])

la_weight = Hyperparameter("la_weight", [0.2, 0.15, 0.1])
forzajuve_weight = Hyperparameter("forzajuve_weight", [0.8, 0.9, 0.85])

final_callable = Callable(obj=Hybrid, args=[
    (la_callable, la_weight),
    (forzajuve_callable, forzajuve_weight)
])


HyperparameterTuner(final_callable, cartesian_product=False).run()
