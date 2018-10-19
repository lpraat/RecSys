import numpy as np
import scipy.sparse as sp
import sklearn.metrics as metrics
import pandas
from timeit import default_timer as timer

from src.const import NUM_PLAYLIST, NUM_TRACKS
from src.metrics import evaluate
from src.writer import create_submission
from src.alg.item_knn import ItemKNN
from src.data import Cache, save_file, load_file

# Run item KNN
recsys = ItemKNN()
recsys.dataset = "interactions"
preds = recsys.run(range(NUM_PLAYLIST))
create_submission("itemknn", preds)


""" preds = []
for pi in range(NUM_PLAYLIST):
    # Get rankings for this playlist
    # and filter out already added tracks
    mask    = 1 - data.getrow(pi).toarray().squeeze()
    ranking = rankings.getrow(pi).toarray().squeeze() * mask
    top = (-ranking).argsort()[:10]
    preds.append([pi, ] + top.tolist()) """