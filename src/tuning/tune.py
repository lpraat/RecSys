from src.alg import Ensemble, ItemKNN
from src.tuning.tuner import Hyperparameter, Callable, HyperparameterTuner


# TODO this like src/run.py should go in the .gitignore
# Usage example
# I want to tune alpha and h for the following model
# Ensemble((ItemKNN(("artist_set", 0.1, {}), ("album_set", 0.2, {}), alpha=0.56, h=2.5), 1))

alpha = Hyperparameter("alpha", [0.56])
h = Hyperparameter("h", [2.5])

ensemble = Callable(
    obj=Ensemble,
    args=[
        (Callable(
            obj=ItemKNN,
            args=[("artist_set", 0.1, {}), ("album_set", 0.2, {})],
            kwargs={"alpha": alpha, "h": h}
        ), 1)
    ],
    kwargs=None
)

tuner = HyperparameterTuner(ensemble, cartesian_product=True)
tuner.run()
