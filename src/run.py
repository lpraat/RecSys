import src.alg as alg
from src.const import NUM_PLAYLIST, NUM_TRACKS, NUM_ALBUMS
from src.metrics import evaluate
from src.writer import create_submission
from src.data import Cache, save_file, load_file

# Run item KNN
recsys = alg.ContentKNN(dataset = "train_set", features = [("album_set", 1.5), ("artist_set", 0.2)], h = 0, alpha = 0.5)
preds = recsys.run(range(NUM_PLAYLIST))


""" preds = []
for pi in range(NUM_PLAYLIST):
    # Get rankings for this playlist
    # and filter out already added tracks
    mask    = 1 - data.getrow(pi).toarray().squeeze()
    ranking = rankings.getrow(pi).toarray().squeeze() * mask
    top = (-ranking).argsort()[:10]
    preds.append([pi, ] + top.tolist()) """