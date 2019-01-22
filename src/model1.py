from src.alg import ItemKNN, Hybrid, UserKNN
from src.alg.als import ALS
from src.alg.hybrid_similarity import HybridSimilarity
from src.alg.light import Light
from src.alg.slim import Slim
from src.data import Cache
from src.writer import create_submission

cache = Cache()

slim = Slim(lambda_i=0.025, lambda_j=0.025, all_dataset=True, epochs=3, lr=0.1)
l = Light(no_components=300, epochs=30, loss='warp')
a = ALS(factors=1024, iterations=2)
h1 = HybridSimilarity((ItemKNN(("artist_set", 0.1, {}), ("album_set", 0.2, {})), 0.7),(slim, 0.3))
hf = Hybrid((h1, 0.85), (UserKNN(knn=190), 0.15))
f = Hybrid((hf, 0.8), (Hybrid((l, 0.5), (a, 0.5)), 0.2))

preds = f.run(dataset=cache.fetch("interactions"), targets=cache.fetch("targets"))
create_submission("first", preds)
