from src.alg import ItemKNN, Ensemble, UserKNN, ContentKNN

ItemKNN(
    ("artist_set", 0.08, {}),
    ("album_set", 0.09, {}),
    alpha=0.56
).evaluate()