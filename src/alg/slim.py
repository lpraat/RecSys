import numpy as np

from src.alg.recsys import RecSys


class BPRSampler:
    def __init__(self, urm):
        self.urm = urm.tocsr()
        self.lil = urm.tolil()

    def sample(self):

        # Get a random user and its interactions
        user = np.random.choice(self.urm.shape[0])
        user_interactions = self.urm.indices[self.urm.indptr[user]:self.urm.indptr[user + 1]]

        # Sample a positive interaction
        positive_item = np.random.choice(user_interactions)

        # Sample a negative interaction
        negative_item = np.random.choice(self.urm.shape[1])

        # Use lil fast access property to speed up this re-try
        while self.lil[user, negative_item] == 1:
            negative_item = np.random.choice(self.urm.shape[1])

        return np.array([user, positive_item, negative_item], dtype=np.uint16)

    def sample_batch(self, batch_size):

        for i in range(batch_size):
            pass


# TODO you were right you need to transpose the similarity matrix you get from slim
# so that row turns into column and you can do dot product URM * s


class Slim(RecSys):

    def __init__(self):
        super().__init__()

    def rate(self, dataset=None):
        pass

    def train(self):
        # For i in num_epochs
        # Get sample
        # Predict input
        # Get error from targets
        # Update weights using mgd
        pass

    def predict(self):
        # Get prediction given input
        pass

    def mgd(self):
        # Mini batch gradient descent
        # todo find a way to generalize to any batch size
        pass
