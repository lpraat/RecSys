from src.alg import Ensemble, ItemKNN
from src.tuning.tuner import Hyperparameter, Callable, hyperparameter_search

alpha = Hyperparameter("alpha", [0.1, 0.2, 0.3])

ensemble = Callable(
    obj=Ensemble,
    args=[
        (Callable(
            obj=ItemKNN,
            args=[("artist_set", 0.1, {}), ("album_set", 0.2, {})],
            kwargs={"alpha": alpha, "h": 2.5}
        ), 0.4)
    ],
    kwargs=None
)

hyperparameters = [alpha]

hyperparameter_search(ensemble, hyperparameters)