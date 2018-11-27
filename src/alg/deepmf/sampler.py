import numpy as np
import tensorflow as tf

class DeepMFSampler():

    def __init__(self, urm, num_negative_samples=1, batch_size=1):

        self.batch_size = batch_size
        self.num_negative_samples = num_negative_samples
        self.urm = urm.tocsr()
        self.lil_urm = urm.tolil()

        self.num_users = self.urm.shape[0]
        self.num_items = self.urm.shape[1]

    def build_dataset(self):
        dataset_size = self.urm.nnz * (1 + self.num_negative_samples)

        # Using unsigned int here gives problem when iterating over the tf.data.Dataset
        users = np.empty((dataset_size, 1), dtype=np.int32)
        items = np.empty((dataset_size, 1), dtype=np.int32)
        labels = np.empty((dataset_size, 1), dtype=np.int8)
        curr = 0

        for _ in range(self.urm.nnz):

            # Get a random user and its interactions
            user = np.random.choice(self.urm.shape[0])
            user_interactions = self.urm.indices[self.urm.indptr[user]:self.urm.indptr[user + 1]]

            # Sample a positive interaction
            positive_item = np.random.choice(user_interactions)
            users[curr, :] = [user]
            items[curr, :] = [positive_item]
            labels[curr, :] = [1]
            curr += 1

            # Sample negative interactions
            for _ in range(self.num_negative_samples):
                negative_item = np.random.choice(self.urm.shape[1])

                # Use lil fast access property to speed up this re-try
                while self.lil_urm[user, negative_item] == 1:
                    negative_item = np.random.choice(self.urm.shape[1])

                users[curr, :] = [user]
                items[curr, :] = [negative_item]
                labels[curr, :] = [0]
                curr += 1

        return tf.data.Dataset.from_tensor_slices((users, items, labels))