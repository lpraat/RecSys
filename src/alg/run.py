from src.alg import ItemKNN, UserClusterize
from src.alg.hybrid_similarity import HybridSimilarity
from src.alg.p3alpha import P3Alpha
from src.alg.rp3beta import RP3Beta
from src.alg.smart_ensemble.smart_ensemble import SmartEnsemble
from src.alg.smart_ensemble.utils import write_preds_to_file

hybrid_graph = HybridSimilarity((P3Alpha(knn=1024, alpha=1), 0.15), (RP3Beta(knn=1024, alpha=1, beta=0.5), 0.95))
hybrid_graph.evaluate()
ItemKNN().evaluate()


