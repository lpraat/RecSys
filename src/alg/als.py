import numpy as np
import scipy.sparse as sp
import multiprocessing as mp

from src.alg.recsys import RecSys

from implicit.als import AlternatingLeastSquares as als


class ALS(RecSys):

    def __init__(self, factors=10, iterations=10, reg=0.01, use_gpu=False, num_threads=mp.cpu_count()):
        super().__init__()
        self.factors = factors
        self.iterations = iterations
        self.reg = reg
        self.use_gpu = use_gpu
        self.num_threads = num_threads
        self.model = als(factors=self.factors,
                         iterations=self.iterations,
                         regularization=self.reg,
                         use_gpu=self.use_gpu,
                         num_threads=self.num_threads)

    def compute_similarity(self, dataset):
        raise NotImplementedError

    def rate(self, dataset, targets):
        print("training als ...")
        self.model.fit(dataset.T)

        print("computing ratings ...")
        ratings = np.empty((len(targets), dataset.shape[1]), dtype=np.float32)
        for i, target in enumerate(targets):
            if i % 1000 == 0:
                print(f"computed ratings for {i} playlists")

            r = self.model.recommend(userid=target, user_items=dataset, filter_already_liked_items=False, N=1000)

            items = []
            rates = []
            for item_id, rating in r:
                items.append(item_id)
                rates.append(rating)

            new_row = np.zeros((1, dataset.shape[1]), dtype=np.float32)
            new_row[0, items] = rates
            ratings[i] = new_row

        return sp.csr_matrix(ratings, dtype=np.float32)







