import random

from src.const import NUM_TRACKS
from src.data import Cache
from src.metrics import leave_one_out
from src.alg.item_based import dok_matrix_to_sparse_tensor

cache = Cache()

print(dok_matrix_to_sparse_tensor(cache.fetch("interactions")))