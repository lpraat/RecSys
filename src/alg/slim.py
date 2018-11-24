import numpy as np
import time

from src.alg import ItemKNN
from src.alg.bpr import BPRSampler
from src.alg.recsys import RecSys
from numpy.linalg import linalg as LA
import scipy.sparse as sp

from src.alg.utils import knn


class Slim(RecSys):
    def __init__(self, all_dataset, lr=0.01, batch_size=1, epochs=1, lambda_i=0, lambda_j=0, knn=np.inf, dual=False):
        super().__init__()

        self.dual = dual
        self.all_dataset = all_dataset

        self.lambda_i = lambda_i
        self.lambda_j = lambda_j

        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs

        self.knn = knn

        self.num_interactions = None
        self.bpr_sampler = None


    def compute_similarity(self, dataset):
        if self.all_dataset:
            urm = self.cache.fetch("interactions")
        else:
            urm = self.cache.fetch("train_set")

        urm = urm.tocsr()

        self.num_interactions = urm.nnz
        urm = urm.T if self.dual else urm

        urm = sp.csr_matrix(urm)
        self.bpr_sampler = BPRSampler(urm)

        slim_dim = urm.shape[1]
        # Similarity matrix slim is trying to learn
        s = np.zeros((slim_dim, slim_dim), dtype=np.float32)

        print("Training Slim...")
        start = time.time()
        self.train(self.lr, self.batch_size, self.epochs, urm, s)
        print("elapsed: {:.3f}s\n".format(time.time() - start))

        print("Taking Slim k nearest neighbors...")
        start = time.time()
        s = knn(s.T, knn=self.knn)
        print("elapsed: {:.3f}s\n".format(time.time() - start))

        return s

    def rate(self, dataset):

        s = self.compute_similarity(dataset)

        print("Computing Slim ratings...")
        start = time.time()
        if self.dual:
            ratings = (dataset.T * s).tocsr()
        else:
            ratings = (dataset * s).tocsr()
        print("elapsed: {:.3f}s\n".format(time.time() - start))
        del s

        if self.dual:
            return ratings.T
        else:
            return ratings

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

    def sgd(self, lr, num_epochs, urm, slim_matrix):

        for i in range(num_epochs):

            print(f"Sampling for epoch {i+1}")
            batches = self.build_batches(1)
            print(f"Started epoch {i+1}")

            for batch in batches:
                u = batch[0][0]
                i = batch[0][1]
                j = batch[0][2]

                user_indices = urm.indices[urm.indptr[u]:urm.indptr[u + 1]]

                # Get current prediction
                x_ui = slim_matrix[i, user_indices]
                x_uj = slim_matrix[j, user_indices]

                x_uij = np.sum(x_ui - x_uj)

                # Compute gradient of log(sigmoid(x_uij))
                # gradient = expit(-x_uij)
                gradient = 1 / (1 + np.exp(x_uij))

                # Get current loss
                # loss = self.loss(x_ui, x_uj, x_uij)
                # print(f"Current loss is {loss}")

                # Update i parameters
                slim_matrix[i, user_indices] -= lr * (
                    - gradient + (self.lambda_i * slim_matrix[i, user_indices]))
                slim_matrix[i, i] = 0

                # Update j parameters
                slim_matrix[j, user_indices] -= lr * (
                    gradient + (self.lambda_j * slim_matrix[j, user_indices]))
                slim_matrix[j, j] = 0

    def train(self, lr, batch_size, num_epochs, urm, slim_matrix):
        if batch_size == 1:
            self.sgd(lr, num_epochs, urm, slim_matrix)
        else:
            self.mgd(lr, batch_size, num_epochs, urm, slim_matrix)

    def mgd(self, lr, batch_size, num_epochs, urm, slim_matrix):

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
                x_ui = urm[u, :].multiply(slim_matrix[i])
                x_uj = urm[u, :].multiply(slim_matrix[j])
                x_uij = np.sum(x_ui - x_uj, axis=1)

                # Get current loss
                # loss = self.loss(i_param, j_param, x_uij)
                # print(f"Current loss is {loss}")

                # Compute gradient of 1/m * log(sigmoid(x_uij))
                gradient = np.sum(1 / (1 + np.exp(x_uij))) / m

                # Update only items corresponding to users in this batch
                items_mask = urm[u, :].A > 0

                # Compute gradients and update parameters
                # Update i parameters
                slim_matrix[i] -= lr * (-gradient + ((self.lambda_i / m) * slim_matrix[i])) * items_mask
                slim_matrix[i, i] = 0

                # Update j parameters
                slim_matrix[j] -= lr * (gradient + ((self.lambda_j / m) * slim_matrix[j])) * items_mask
                slim_matrix[j, j] = 0

    def loss(self, i_param, j_param, x_uij):
        m = x_uij.shape[0]
        loss = - (1 / m) * np.sum(np.log(1 / (1 + np.exp(-x_uij))))
        reg = (0.5 / m) * (self.lambda_i * (LA.norm(i_param) ** 2) + self.lambda_j * (LA.norm(j_param) ** 2))

        return loss + reg
