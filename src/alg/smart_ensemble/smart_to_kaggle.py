from src.alg import ItemKNN, Hybrid, UserKNN
from src.alg.hybrid_similarity import HybridSimilarity
from src.alg.p3alpha import P3Alpha
from src.alg.rp3beta import RP3Beta
from src.alg.slim import Slim
from src.alg.smart_ensemble.utils import write_preds_to_file, build_preds_from_file
from src.alg.smart_ensemble.write_max import build_max_preds
from src.data import Cache
from src.metrics import evaluate

cache = Cache()
# Write here your models remember to using the WHOLE dataset
# They will be put in the smart_ensemble/preds folder
# Use the write_to_file utility in this package

# HYBRID GRAPH
hybrid_graph = HybridSimilarity((P3Alpha(knn=1024, alpha=1), 0.15),
                                        (RP3Beta(knn=1024, alpha=1, beta=0.5), 0.95))

write_preds_to_file("hybrid_graph", hybrid_graph, dataset="interactions", targets=cache.fetch("targets"))
map = evaluate(build_preds_from_file("hybrid_graph", test=False), cache.fetch("test_set"))
print(map)
assert(map == 0)


# HYBRID ITEM USER
hybrid_item_user = Hybrid((ItemKNN(("artist_set", 0.1, {}), ("album_set", 0.2, {})), 0.3),
                                  (UserKNN(knn=200), 0.2))

write_preds_to_file("hybrid_item_user", hybrid_item_user, dataset="interactions", targets=cache.fetch("targets"))
map = evaluate(build_preds_from_file("hybrid_item_user", test=False), cache.fetch("test_set"))
print(map)
assert(map == 0)


# DOUBLE SLIM
double_slim = HybridSimilarity((Slim(epochs=1, lr=0.1, lambda_i=0.01, all_dataset=True), 0.5),
                                       (Slim(epochs=1, lr=0.1, lambda_j=0.01, all_dataset=True), 0.5))

write_preds_to_file("double_slim",double_slim, dataset="interactions", targets=cache.fetch("targets"))
map = evaluate(build_preds_from_file("double_slim", test=False), cache.fetch("test_set"))
print(map)
assert(map == 0)

# SLIM
slim = Slim(epochs=3, lr=0.1, lambda_i=0.025, lambda_j=0.025, all_dataset=True)
write_preds_to_file("slim",slim, dataset="interactions", targets=cache.fetch("targets"))
map = evaluate(build_preds_from_file("slim", test=False), cache.fetch("test_set"))
print(map)
assert(map == 0)

## FORZAJUVE
h1 = HybridSimilarity((ItemKNN(("artist_set", 0.1, {}), ("album_set", 0.2, {})), 0.7),
                              (Slim(lambda_i=0.025, lambda_j=0.025, all_dataset=True, epochs=3, lr=0.1), 0.3))
m1 = Hybrid((h1, 0.85), (UserKNN(knn=190), 0.15))

write_preds_to_file("forzajuve", m1, dataset="interactions", targets=cache.fetch("targets"))
map = evaluate(build_preds_from_file("forza_juve", test=False), cache.fetch("test_set"))
print(map)
assert(map == 0)


# Use smart ensembler from file
preds = build_max_preds(["hybrid_item_user", "hybrid_graph", "double_slim", "slim", "forzajuve"], test=False)

# Evaluate
map = evaluate(preds, cache.fetch("test_set"))
print(map)
assert(map == 0)


# Assert all model evaluation go to false
# this is a good indicator you've used all the dataset :D