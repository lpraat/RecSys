import numpy as np
import scipy.sparse as sp
import pandas
from timeit import default_timer as timer

from src.const import NUM_PLAYLIST, NUM_TRACKS
from src.data import Cache
from src.metrics import leave_one_out
from src.writer import create_submission
from src.alg.item_based import get_rankings

# Create cache and fetch train set
cache = Cache()
data    = cache.fetch("train_set").tocsr()
albums  = cache.fetch("album_set").tocsr()
artists = cache.fetch("artist_set").tocsr()
test    = cache.fetch("test_set")

rankings = get_rankings(data, albums, artists, weights = [1, 0, 0], normalize = [False, False, False]).tocsr()

# Calc 10 predictions
print("computing predictions ...\n")

preds = []
for pi in range(NUM_PLAYLIST):
    # Get rankings for this playlist
    # and filter out already added tracks
    mask    = 1 - data.getrow(pi).toarray().squeeze()
    ranking = rankings.getrow(pi).toarray().squeeze() * mask
    top = (-ranking).argsort()[:10]
    preds.append([pi, ] + top.tolist())