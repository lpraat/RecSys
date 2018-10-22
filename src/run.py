import src.alg as alg
from src.const import NUM_PLAYLIST
from src.data import Cache, save_file, load_file
import multiprocessing.pool

# Run item KNN
recsys = alg.ItemKNN(alpha = 0.65, h = 3, qfunc=lambda x: x ** 1.1)
preds = recsys.run(range(NUM_PLAYLIST))