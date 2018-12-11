from timeit import default_timer as timer

from src.alg.recsys import RecSys
import similaripy as sim


class ItemKNNPlus(RecSys):

    def __init__(self, t1=1, t2=1, l=0.5, c=0.5, knn=1000):
        super().__init__()
        self.t1 = t1
        self.t2 = t2
        self.l = l
        self.c = c
        self.knn = knn

    def compute_similarity(self, dataset=None):
        print("computing splus similarity ...")
        return sim.cosine(dataset.T, k=self.knn, verbose=True)

    def rate(self, dataset):
        s = self.compute_similarity(dataset)
        print("computing ratings ...")
        start = timer()
        # Compute playlist-track ratings
        ratings = (dataset * s).tocsr()
        print("elapsed: {:.3f}s\n".format(timer() - start))
        del s

        return ratings




