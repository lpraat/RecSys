import implicit
import time

from src.alg.recsys import RecSys


class BM25(RecSys):
    def __init__(self, knn=1000, k1=1.2, b=0.75):
        super().__init__()
        self.knn = knn
        self.k1 = k1
        self.b = b

        self.model = implicit.nearest_neighbours.BM25Recommender(K=self.knn, K1=self.k1, B=self.b)

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


