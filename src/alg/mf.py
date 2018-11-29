import numpy as np
import scipy.sparse as sp

from src.alg.bpr import BPRSampler
from src.alg.recsys import RecSys


# TODO this method works really really bad, remove it?
class MFBpr(RecSys):
    def __init__(self, lr=0.01, batch_size=1, epochs=1, all_dataset=True, factors=8, lambda_u=0, lambda_i=0, lambda_j=0):

        super().__init__()

        self.lambda_u = lambda_u
        self.lambda_i = lambda_i
        self.lambda_j = lambda_j

        self.batch_size = batch_size

        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = epochs

        self.num_interactions = None
        self.bpr_sampler = None

        self.all_dataset = all_dataset

        self.factors = factors

        raise DeprecationWarning("This method has not been updated")

    def build_batches(self, batch_size):
        assert batch_size <= self.num_interactions, "Batch size is too big"
        batches = []
        full_batches = self.num_interactions // batch_size

        for _ in range(full_batches):
            batches.append(self.bpr_sampler.sample_batch(batch_size))

        need_fill = self.num_interactions % batch_size
        if need_fill:
            batches.append(self.bpr_sampler.sample_batch(need_fill))

        return batches

    def rate(self, dataset=None):

        if self.all_dataset:
            urm = self.cache.fetch("interactions")
        else:
            urm = self.cache.fetch("train_set")

        self.num_interactions = urm.nnz
        self.bpr_sampler = BPRSampler(urm)

        num_users = urm.shape[0]
        num_items = urm.shape[1]
        user_factors = np.random.random_sample((num_users, self.factors)).astype(np.float32)
        item_factors = np.random.random_sample((num_items, self.factors)).astype(np.float32)

        self.train(self.lr, self.num_epochs, user_factors, item_factors)

        user_factors = sp.csr_matrix(user_factors)
        item_factors = sp.csr_matrix(item_factors)

        return user_factors * item_factors.T

    def train(self, lr, num_epochs, user_factors, item_factors):

        if self.batch_size == 1:
            self.sgd(lr, num_epochs, user_factors, item_factors)
        else:
            raise NotImplementedError
            # self.mgd(lr, num_epochs, user_factors, item_factors)

    def sgd(self, lr, num_epochs, user_factors, item_factors):
        for i in range(num_epochs):

            print(f"Sampling for epoch {i+1}")
            batches = self.build_batches(1)
            print(f"Started epoch {i+1}")

            for batch in batches:

                u = batch[0][0]
                i = batch[0][1]
                j = batch[0][2]

                x = np.dot(user_factors[u, :], item_factors[i, :] - item_factors[j, :])

                z = 1.0 / (1.0 + np.exp(x))

                u_factors = user_factors[u, :]
                i_factors = item_factors[i, :]
                j_factors = item_factors[j, :]

                # Update u params
                d = (i_factors - j_factors) * z - self.lambda_u * u_factors
                user_factors[u, :] += lr * d

                # Update i params
                d = u_factors * z - self.lambda_i * i_factors
                item_factors[i, :] += lr * d

                # Update j params
                d = -u_factors * z - self.lambda_j * j_factors
                item_factors[j, :] += lr * d

    def mgd(self):
        raise NotImplementedError