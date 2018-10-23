from src.alg import ItemKNN, Ensemble, UserKNN, ContentKNN
from src.const import NUM_PLAYLIST
from src.writer import create_submission
from src.data import Cache
from multiprocessing.pool import Pool
import numpy as np

def eval(h):
    return ItemKNN(h=1., asym=False, qfunc=lambda w: w ** 0.9).evaluate()

""" max = (0., 0.)
for h in np.linspace(0.5, 1.5, 10):
    pool = Pool(3)
    values = np.array([h, h + 1. / 30, h + 2. / 30])
    scores = pool.map(eval, values)
    pool.join
    argmax = np.argmax(scores)
    if scores[argmax] > max[1]:
        max = (values[argmax], scores[argmax])
    print("current max: {}\n".format(max))
print(max)
quit() """
score = Ensemble(models=[
    (ItemKNN(), 1.),
    (UserKNN(), 1.)
]).evaluate()