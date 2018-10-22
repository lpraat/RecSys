from src.alg import ItemKNN, Ensamble
from src.const import NUM_PLAYLIST

# Run item KNN
#recsys = ItemKNN(alpha=0.3, h=0, neighbours=500, qfunc=lambda w: w ** 1.5)
#preds = recsys.run(range(NUM_PLAYLIST))


recsys = Ensamble(
    models = [(ItemKNN(alpha=0.5, h=1700), 0.5),
    (ItemKNN(alpha=0.5, h=100), 0.5)]
).run(range(NUM_PLAYLIST))