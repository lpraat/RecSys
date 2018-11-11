import numpy as np


class BPRSampler:
    def __init__(self, urm):
        self.urm = urm.tocsr()

        # todo maybe using a dict here is better?
        self.lil_urm = urm.tolil()

    def sample(self):

        # Get a random user and its interactions
        user = np.random.choice(self.urm.shape[0])
        user_interactions = self.urm.indices[self.urm.indptr[user]:self.urm.indptr[user + 1]]

        # Sample a positive interaction
        positive_item = np.random.choice(user_interactions)

        # Sample a negative interaction
        negative_item = np.random.choice(self.urm.shape[1])

        # Use lil fast access property to speed up this re-try
        while self.lil_urm[user, negative_item] == 1:
            negative_item = np.random.choice(self.urm.shape[1])

        return np.array([user, positive_item, negative_item], dtype=np.int32)

    def sample_batch(self, batch_size):
        batch = np.zeros((batch_size, 3), dtype=np.uint16)
        for i in range(batch_size):
            batch[i] = self.sample()

        return batch