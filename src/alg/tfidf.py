import implicit
import time

from src.alg.recsys import RecSys


class TfIdf(RecSys):
    def __init__(self, knn=1000):
        super().__init__()
        self.knn = knn
        self.model = implicit.nearest_neighbours.TFIDFRecommender(K=self.knn)

    def compute_similarity(self, dataset):
        self.model.fit(dataset.T)
        return self.model.similarity

    def rate(self, dataset, targets):
        print("Using all dataset " + str(dataset.nnz))

        s = self.compute_similarity(dataset)
        print("computing ratings ...")
        start = time.time()

        print(len(targets))
        # Compute playlist-track ratings
        ratings = (dataset[targets, :] * s).tocsr()
        print("elapsed: {:.3f}s\n".format(time.time() - start))
        return ratings


