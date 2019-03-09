from src.alg.item_knn import ItemKNN
from src.data import Cache
from src.writer import create_submission

cache = Cache()

item_knn = ItemKNN()

preds = item_knn.run(dataset=cache.fetch("interactions"), targets=cache.fetch("targets"))
create_submission("simple_item_knn_predictions", preds)
