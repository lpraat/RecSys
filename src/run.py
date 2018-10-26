import multiprocessing as mp

from src.alg import ItemKNN, Ensemble, UserKNN, ContentKNN, UserClusterize
from src.data import Cache, build_train_set_uniform

# Global cache
cache = Cache()
ItemKNN(h=3).evaluate()
