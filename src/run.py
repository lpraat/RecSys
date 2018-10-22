from src.alg import ItemKNN, Ensemble, UserKNN, ContentKNN
from src.const import NUM_PLAYLIST
from src.writer import create_submission
from src.data import Cache

Ensemble(models=[
    (ItemKNN(alpha=0.56, h=2.68), 1.8),
    (ContentKNN(features=[
        ("artist_set", 0.7, {"alpha": 0.56}),
        ("album_set", 0.3, {"alpha": 0.56})
    ]), 0.2),
    (UserKNN(knn=180), 0.6)
]).evaluate()