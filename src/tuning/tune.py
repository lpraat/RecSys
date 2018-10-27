from src.alg import Ensemble, ItemKNN
from src.tuning.tuner import Hyperparameter, Callable, hyperparameter_search

alpha = Hyperparameter("alpha", [0.1])
h = Hyperparameter("h", [2.5])

ensemble = Callable(
    obj=Ensemble,
    args=[
        (Callable(
            obj=ItemKNN,
            args=[("artist_set", 0.1, {}), ("album_set", 0.2, {})],
            kwargs={"alpha": alpha, "h": h}
        ), 0.4)
    ],
    kwargs=None
)

# TODO this parameter can be removed, since tuner can build it by looking at the model
hyperparameters = [alpha, h]

hyperparameter_search(ensemble, hyperparameters, write_to_file=True, cartesian_product=True)