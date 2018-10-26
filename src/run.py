import multiprocessing as mp

from src.alg import ItemKNN, Ensemble, UserKNN, ContentKNN, UserClusterize
from src.data import Cache, build_train_set_uniform

# Global cache
cache = Cache()

# Model to evaluate
model = Ensemble(
    (ItemKNN(
        ("artist_set", 0.1, {}),
        ("album_set", 0.2, {}),
        alpha=0.56, h=2.5
    ), 0.4),
    (UserKNN(knn=180), 0.2)
)

# Initial values
iter = 10

# Evaluation routine
def eval_routine(i):
    print("worker #{} starting ...\n".format(i))

    # Rebuild train set and test set
    train_set, test_set = build_train_set_uniform(
        cache.fetch("interactions"),
        cache.fetch("targets"),
        0.2
    )

    return model.evaluate(train_set=train_set, test_set=test_set) 

# Thread pool
workers = mp.Pool(2)
scores = workers.map(eval_routine, range(iter))
workers.close()

# Average
print("average score: {}".format(sum(scores)))