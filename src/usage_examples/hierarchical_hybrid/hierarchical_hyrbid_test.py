from src.alg.hybrid import Hybrid
from src.alg.hybrid_similarity import HybridSimilarity
from src.alg.item_knn import ItemKNN
from src.alg.slim import Slim
from src.alg.user_knn import UserKNN
from src.data import Cache
from src.writer import create_submission

cache = Cache()

slim = Slim(lambda_i=0.025, lambda_j=0.025, all_dataset=True, epochs=3, lr=0.1)

# Hybrid similarity between item knn (weighted 0.8) and slim (weighted slim)
h1 = HybridSimilarity([
    (ItemKNN([("artist_set", 0.11, {}), ("album_set", 0.13, {})], normalize=True), 0.8),
    (slim, 0.2)])

# Hyrbid ratings between h1 hybrid (weighted 0.85) and user knn (weighted 0.15)
final_hybrid = Hybrid([
    (h1, 0.85),
    (UserKNN(knn=200), 0.15)])

preds = final_hybrid.run(dataset=cache.fetch("interactions"), targets=cache.fetch("targets"))
create_submission("hierarchical_hybrid_predictions", preds)
