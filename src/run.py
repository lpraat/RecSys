from src.alg import ItemKNN, Ensemble, UserKNN, ContentKNN, UserClusterize

UserClusterize(Ensemble(
    (ItemKNN(
        ("artist_set", 0.05, {}),
        ("album_set", 0.08, {}),
        alpha=0.56, h=2.
    ), 0.1),
    (UserKNN(knn=100), 0.4)
)).evaluate()