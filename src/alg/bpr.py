import numpy as np


class BPRSampler:
    """
    Bayesian personalized ranking sampler
    """
    def __init__(self, urm):
        self.urm = urm.tocsr()
        self.lil_urm = urm.tolil()

    def sample(self):

        while True:
            # Get a random user and its interactions
            user = np.random.choice(self.urm.shape[0])
            user_interactions = self.urm.indices[self.urm.indptr[user]:self.urm.indptr[user + 1]]

            if user_interactions.any():
                # Sample a positive interaction
                positive_item = np.random.choice(user_interactions)

                # Sample a negative interaction
                negative_item = np.random.choice(self.urm.shape[1])

                # Use lil fast access property to speed up this re-try
                while self.lil_urm[user, negative_item] == 1:
                    negative_item = np.random.choice(self.urm.shape[1])

                return np.array([user, positive_item, negative_item], dtype=np.uint32)

    def sample_batch(self, batch_size):
        batch = np.zeros((batch_size, 3), dtype=np.uint32)
        for i in range(batch_size):
            batch[i] = self.sample()

        return batch
