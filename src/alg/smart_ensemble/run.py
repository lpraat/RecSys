from src.alg.smart_ensemble.intialize_sets import initialize_sets
from src.alg.smart_ensemble.smart_validate import validate_and_build_preds

#initialize_sets(5)

validate_and_build_preds(["hybrid_graph", "item_knn"], k=5, mode='borda')
