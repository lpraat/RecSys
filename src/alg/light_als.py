from lightfm import LightFM

import scipy.sparse as sp
import multiprocessing as mp


class LightALS():

    def __init__(self):
        super().__init__()
        pass

    def compute_similarity(self, dataset):
        raise NotImplementedError

    def run(self):
        pass

    def rate(self, dataset):
        pass

class Light():

    def __init__(self, no_components=10, learning_schedule='adagrad', loss='warp', learning_rate=0.05,
                       epochs=1, num_threads=mp.cpu_count()):

        self.no_components = no_components
        self.learning_schedule = learning_schedule
        self.loss = loss
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.num_threads = num_threads

        self.model = LightFM(no_components=self.no_components, learning_schedule=self.learning_schedule, loss=self.loss,
                             learning_rate=self.learning_rate)

    def train(self, dataset):
        self.model.fit(interactions=dataset, epochs=self.epochs, num_threads=self.num_threads, verbose=True)


    def predict(self, user_id):
        pass


def createLight(no_components=10, learning_schedule='adagrad', loss='warp', learning_rate=0.05):
    m = LightFM(loss='warp',
    pass

def createALS():
    pass