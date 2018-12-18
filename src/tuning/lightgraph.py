from src.alg import UserKNN, Hybrid
from src.alg.als import ALS
from src.alg.light import Light
from src.alg.rp3beta import RP3Beta
from src.tuning.tuner import Callable, Hyperparameter, HyperparameterTuner

RP3Beta(alpha=1, beta=0.3, knn=1000)

rp3_knn = Hyperparameter("rp3_knn", [128, 256, 512, 1024])
rp3 = Callable(obj=RP3Beta, kwargs={'alpha':1, 'beta':0.3, 'knn': rp3_knn})

user_knn = Hyperparameter("user_knn", [190, 200])
user = Callable(obj=UserKNN, kwargs={"knn": user_knn})

rp3_weight = Hyperparameter("rp3_weight", [0.9, 0.8, 0.7, 0.6])
user_weight = Hyperparameter("user_weight", [0.1, 0.2, 0.3, 0.4])

h1 = Callable(obj=Hybrid, args=[
    (rp3, rp3_weight),
    (user, user_weight)
])

light = Callable(obj=Light, kwargs={'no_components':300, 'epochs':30})
als = Callable(obj=ALS, kwargs={'factors': 1024, 'iterations': 2})

light_weight = Hyperparameter("light_weight", [0.8, 0.7, 0.6, 0.5, 0.3, 0.2])
als_weight = Hyperparameter("als_weight", [0.8, 0.7, 0.6, 0.5, 0.3, 0.2])

h2 = Callable(obj=Hybrid, args=[
    (light, light_weight),
    (als, als_weight)
])

h1_weight = Hyperparameter("user_rp3_weight", [0.8, 0.7, 0.6, 0.5, 0.3, 0.2])
h2_weight = Hyperparameter("light_als_weight", [0.8, 0.7, 0.6, 0.5, 0.3, 0.2])

final = Callable(obj=Hybrid, args=[
    (h1, h1_weight),
    (h2, h2_weight)
])

HyperparameterTuner(final, cartesian_product=False).run()
