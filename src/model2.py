from src.alg import Hybrid, ItemKNN, UserKNN
from src.alg.als import ALS
from src.alg.hybrid_similarity import HybridSimilarity
from src.alg.light import Light
from src.alg.slim import Slim
from src.data import Cache
from src.writer import create_submission

cache = Cache()

slim = Slim(lambda_i=0.001, lambda_j=0.001, all_dataset=False, epochs=3, lr=0.1)
l = Light(no_components=300, epochs=50, loss='warp')
a = ALS(factors=1024, iterations=2)
h1 = HybridSimilarity((ItemKNN(("artist_set", 0.1, {}), ("album_set", 0.2, {})), 0.65), (slim, 0.35))
hf = Hybrid((h1, 0.9), (UserKNN(knn=200), 0.1))
f = Hybrid((hf, 0.85), (Hybrid((l, 0.7), (a, 0.3)), 0.15))

preds = f.run(dataset=cache.fetch("interactions"), targets=cache.fetch("targets"))
create_submission("second", preds)
