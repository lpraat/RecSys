from lightfm import LightFM

import numpy as np
import scipy.sparse as sp
import multiprocessing as mp

from src.alg.recsys import RecSys
from src.const import NUM_TRACKS


class Light(RecSys):
    def __init__(self, no_components=10, learning_schedule='adagrad',
                 loss='warp', learning_rate=0.05, epochs=1, num_threads=mp.cpu_count()):

        super().__init__()
        self.no_components = no_components
        self.learning_schedule = learning_schedule
        self.loss = loss
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.num_threads = num_threads

        self.model = LightFM(no_components=self.no_components,
                             learning_schedule=self.learning_schedule,
                             loss=self.loss,
                             learning_rate=self.learning_rate)

        self.cache = False

    def compute_similarity(self, dataset):
        raise NotImplementedError

    def rate(self, dataset, targets):

        print("Using all dataset " + str(dataset.nnz))

        if not self.cache:
            print("training light ...")
            self.model.fit(interactions=dataset,
                           epochs=self.epochs,
                           num_threads=self.num_threads,
                           verbose=True)
            self.cache = True
        else:
            print("light already trained, using cache ...")

        print("computing ratings ...")
        ratings = np.empty((len(targets), dataset.shape[1]), dtype=np.float32)
        tracks = [i for i in range(NUM_TRACKS)]
        for i, target in enumerate(targets):
            if i % 1000 == 0:
                print(f"computed ratings for {i} playlists")
            new_row = self.model.predict(target, tracks)
            discard = np.argpartition(new_row, -1000)[:-1000]
            new_row[discard] = 0
            ratings[i] = new_row

        return sp.csr_matrix(ratings, dtype=np.float32)