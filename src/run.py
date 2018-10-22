from src.alg import ItemKNN, Ensemble, UserKNN
from src.const import NUM_PLAYLIST
from src.writer import create_submission
from src.data import Cache

# Run item KNN
#recsys = ItemKNN(alpha=0.5, h=2.7)
#preds = recsys.run(range(NUM_PLAYLIST))
#quit()

cache = Cache()

preds = Ensemble(dataset="interactions", models=[
    (ItemKNN(), 0.8),
    (UserKNN(neighbours=100), 0.2)
]).run(cache.fetch("targets"))
create_submission("item-user-knn", preds)