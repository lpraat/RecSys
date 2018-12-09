from src.alg.smart_ensemble.initialize_sets import initialize_sets
from src.alg.smart_ensemble.smart_validate import validate_and_build_preds

initialize_sets(10)

#validate_and_build_preds(["forzajuve", "hybrid_graph", "double_slim", "hybrid_item_user", "slim"],  k=5, mode='borda')
#validate_and_build_preds(["forzajuve", "hybrid_graph", "double_slim", "hybrid_item_user", "slim"],  k=5, mode='max')
