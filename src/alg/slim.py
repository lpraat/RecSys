import numpy as np
import time

from src.alg import ItemKNN
from src.alg.bpr import BPRSampler
from src.alg.recsys import RecSys
from numpy.linalg import linalg as LA
from scipy.special import expit

from src.alg.utils import knn


class Slim(RecSys):
    def __init__(self, all_dataset, lr=0.01, batch_size=1, epochs=1, lambda_i=0, lambda_j=0, knn=100, dual=False):
        super().__init__()

        self.dual = dual

        if all_dataset:
            urm = self.cache.fetch("interactions")
        else:
            urm = self.cache.fetch("train_set")

        self.urm = urm.tocsr()
        self.num_interactions = self.urm.nnz
        self.urm = self.urm.T if self.dual else self.urm

        slim_dim = self.urm.shape[1]
        self.slim_matrix = np.zeros((slim_dim, slim_dim), dtype=np.float32)

        self.lambda_i = lambda_i
        self.lambda_j = lambda_j

        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs

        self.knn = knn

        # Similarity matrix slim is trying to learn
        self.bpr_sampler = BPRSampler(self.urm)

    def rate(self, dataset=None):

        print("Training Slim")
        start = time.time()
        self.train(self.lr, self.batch_size, self.epochs)
        print("elapsed: {:.3f}s\n".format(time.time() - start))

        print("Taking Slim k nearest neighbors")
        start = time.time()
        knn_slim_similarity = knn(self.slim_matrix)

        print("elapsed: {:.3f}s\n".format(time.time() - start))


        print("Computing Slim ratings")
        start = time.time()

        if self.dual:
            ratings = (dataset.T * knn_slim_similarity).tocsr()
        else:
            ratings = (dataset * knn_slim_similarity).tocsr()
        print("elapsed: {:.3f}s\n".format(time.time() - start))

        # todo if dual
        return ratings.T

    def build_batches(self, batch_size):
        assert batch_size <= self.num_interactions, "Batch size is too big"
        batches = []
        full_batches = self.num_interactions // batch_size

        for i in range(full_batches):
            batches.append(self.bpr_sampler.sample_batch(batch_size))

        need_fill = self.num_interactions % batch_size
        if need_fill:
            batches.append(self.bpr_sampler.sample_batch(need_fill))

        return batches

    def sgd(self, lr, num_epochs):

        for i in range(num_epochs):

            print(f"Sampling for epoch {i+1}")
            batches = self.build_batches(1)

            print(batches[0])
            print(f"Started epoch {i+1}")

            for batch in batches:
                # t = time.time()
                u = batch[0][0]
                i = batch[0][1]
                j = batch[0][2]

                user_indices = self.urm.indices[self.urm.indptr[u]:self.urm.indptr[u + 1]]

                # Get current prediction
                x_ui = self.slim_matrix[i, user_indices]
                x_uj = self.slim_matrix[j, user_indices]

                x_uij = np.sum(x_ui - x_uj)

                # Compute gradient of log(sigmoid(x_uij))
                # Use scipy expit to avoid overflows
                gradient = 1 / (1 + expit(x_uij))

                # Get current loss
                # loss = self.loss(x_ui, x_uj, x_uij)
                # print(f"Current loss is {loss}")

                # Update i parameters
                self.slim_matrix[i, user_indices] -= lr * (
                    - gradient + (self.lambda_i * self.slim_matrix[i, user_indices]))
                self.slim_matrix[i, i] = 0

                # Update j parameters
                self.slim_matrix[j, user_indices] -= lr * (
                    gradient + (self.lambda_j * self.slim_matrix[j, user_indices]))
                self.slim_matrix[j, j] = 0
                # print("TIme " + str(time.time() - t))

    def train(self, lr, batch_size, num_epochs):
        if batch_size == 1:
            self.sgd(lr, num_epochs)
        else:
            self.mgd(lr, batch_size, num_epochs)

    def mgd(self, lr, batch_size, num_epochs):

        for i in range(num_epochs):

            print(f"Sampling for epoch {i+1}")
            batches = self.build_batches(batch_size)
            print(f"Started epoch {i+1}")

            for batch in batches:
                u = batch[:, 0]
                i = batch[:, 1]
                j = batch[:, 2]
                m = batch.shape[0]

                # To compute loss
                # u_i_s = {}
                # u_j_s = {}
                # for sample in batch:
                #     uu, ii, jj = sample
                #     if ii not in u_i_s:
                #         u_i_s.update({uu: ii})
                #     if jj not in u_j_s:
                #         u_j_s.update({uu: jj})
                #
                # u_i = list(u_i_s.keys())
                # i_u = list(u_i_s.values())
                # u_j = list(u_j_s.keys())
                # j_u = list(u_j_s.values())
                #
                # i_param = self.urm[u_i].multiply(self.slim_matrix[i_u]).A
                # j_param = self.urm[u_j].multiply(self.slim_matrix[j_u]).A

                # Get current prediction
                x_ui = self.urm[u, :].multiply(self.slim_matrix[i])
                x_uj = self.urm[u, :].multiply(self.slim_matrix[j])
                x_uij = np.sum(x_ui - x_uj, axis=1)

                # Get current loss
                # loss = self.loss(i_param, j_param, x_uij)
                # print(f"Current loss is {loss}")

                # Compute gradient of 1/m * log(sigmoid(x_uij))
                gradient = np.sum(1 / (1 + np.exp(x_uij))) / m

                # Update only items corresponding to users in this batch
                items_mask = self.urm[u, :].A > 0

                # Compute gradients and update parameters
                # Update i parameters
                self.slim_matrix[i] -= lr * (-gradient + ((self.lambda_i / m) * self.slim_matrix[i])) * items_mask
                self.slim_matrix[i, i] = 0

                # Update j parameters
                self.slim_matrix[j] -= lr * (gradient + ((self.lambda_j / m) * self.slim_matrix[j])) * items_mask
                self.slim_matrix[j, j] = 0

    def loss(self, i_param, j_param, x_uij):
#        m = x_uij.shape[0]
        loss = - (1 / m) * np.sum(np.log(1 / (1 + np.exp(-x_uij))))
        reg = (0.5 / m) * (self.lambda_i * (LA.norm(i_param) ** 2) + self.lambda_j * (LA.norm(j_param) ** 2))

        return loss + reg
