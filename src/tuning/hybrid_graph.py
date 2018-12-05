import numpy as np

from src.alg.hybrid_similarity import HybridSimilarity
from src.alg.p3alpha import P3Alpha
from src.alg.rp3beta import RP3Beta
from src.tuning.tuner import Callable, Hyperparameter, HyperparameterTuner

palpha_callable = Callable(P3Alpha, [], {"knn": 1024, "alpha":1})
palpha_weight = Hyperparameter("palpha_weight", np.arange(0.1, 1, 0.05))

rbeta_alpha = Hyperparameter("rbeta_alpha", [1, 2, 3, 4, 5])
rbeta_beta = Hyperparameter("rbeta_beta", [0.3, 0.5, 0.8, 1, 1.5, 2])
rbeta_callable = Callable(RP3Beta, [], {"knn": 1024, "alpha": rbeta_alpha, "beta": rbeta_beta})
rbeta_weight = Hyperparameter("rbeta_weight", np.arange(0.1, 1, 0.05))

hybrid_callable = Callable(obj=HybridSimilarity, args=[
    (palpha_callable, palpha_weight),
    (rbeta_callable, rbeta_weight)
])

HyperparameterTuner(hybrid_callable, cartesian_product=False).run()