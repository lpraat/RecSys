from src.alg import Hybrid, ItemKNN, UserKNN
from src.alg.hybrid_similarity import HybridSimilarity
from src.alg.p3alpha import P3Alpha
from src.alg.rp3beta import RP3Beta
from src.alg.slim import Slim
from src.alg.smart_ensemble.utils import write_preds_to_file

hybrid_graph = HybridSimilarity((P3Alpha(knn=1024, alpha=1), 0.15), (RP3Beta(knn=1024, alpha=1, beta=0.5), 0.95))
write_preds_to_file(name="hybrid_graph", model=hybrid_graph, dataset="train_set")

double_slim = HybridSimilarity((Slim(epochs=1, lr=0.1, lambda_i=0.01, all_dataset=False), 0.5),
                               (Slim(epochs=1, lr=0.1, lambda_j=0.01, all_dataset=False), 0.5))
write_preds_to_file(name="double_slim", model=double_slim, dataset="train_set")

slim = Slim(epochs=3, lr=0.1, lambda_i=0.025, lambda_j=0.025, all_dataset=False)
write_preds_to_file(name="full_slim", model=slim, dataset="train_set")

hybrid_item_user = Hybrid((ItemKNN(("artist_set", 0.1, {}),("album_set", 0.2, {})),0.3),(UserKNN(knn=200), 0.2))
write_preds_to_file(name="hybrid_item_user", model=slim, dataset="train_set")