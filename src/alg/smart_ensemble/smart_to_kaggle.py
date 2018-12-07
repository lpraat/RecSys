from src.alg import ItemKNN
from src.alg.hybrid_similarity import HybridSimilarity
from src.alg.p3alpha import P3Alpha
from src.alg.rp3beta import RP3Beta
from src.alg.smart_ensemble.utils import write_preds_to_file
from src.alg.smart_ensemble.write_borda import build_borda_preds
from src.alg.smart_ensemble.write_max import build_max_preds
from src.data import Cache
from src.metrics import evaluate

cache = Cache()
# Write here your models remember to using the WHOLE dataset
# They will be put in the smart_ensemble/preds folder
# Use the write_to_file utility in this package

write_preds_to_file("item_knn", ItemKNN(), dataset="interactions", targets=cache.fetch("targets"))

hybrid_graph = HybridSimilarity((P3Alpha(knn=1024, alpha=1), 0.15),
                                        (RP3Beta(knn=1024, alpha=1, beta=0.5), 0.95))

write_preds_to_file("hybrid_graph",hybrid_graph, dataset="interactions", targets=cache.fetch("targets"))

# Use smart ensembler from file
preds = build_borda_preds(["item_knn", "hybrid_graph"], test=False)

# Evaluate
map = evaluate(preds, cache.fetch("test_set"))
print(map)
assert(map == 0)


# Assert all model evaluation go to false
# this is a good indicator you've used all the dataset :D