import numpy as np
import time

from src.alg.bpr import BPRSampler
from src.alg.recsys import RecSys
from numpy.linalg import linalg as LA


class Slim(RecSys):
    def __init__(self, lambda_i, lambda_j):
        super().__init__()
        self.urm = self.cache.fetch("train_set").tocsr()
        self.num_interactions = self.urm.nnz
        self.num_users = self.urm.shape[0]
        self.num_items = self.urm.shape[1]

        self.lambda_i = lambda_i
        self.lambda_j = lambda_j

        # Similarity matrix slim is trying to learn
        self.slim_matrix = np.zeros((self.num_items, self.num_items), dtype=np.float32)
        self.mask = np.full((self.num_items, self.num_items), False)

        self.bpr_sampler = BPRSampler(self.urm)

    def rate(self, dataset=None):
        # TODO remember to transpose Slim matrix before computing predictions
        pass

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

    def train(self, lr, batch_size, num_epochs):

        # TODO
        # if batch size == 1 then sgd
        # else                    mgd
        for i in range(num_epochs):
            batches = self.build_batches(batch_size)

            for batch in batches:
                # TODO remove this
                t = time.time()
                u = batch[:, 0]
                i = batch[:, 1]
                j = batch[:, 2]
                m = batch.shape[0]

                # Get current prediction

                # TODO benchmark this vs the other for different small batch sizes

                # This is slower for larger batch size > 64
                '''
                x_ui = self.slim_matrix[i]
                x_uj = self.slim_matrix[j]
                x_uij = x_ui - x_uj
                x_uij = self.urm[u, :].dot(x_uij.T).diagonal()
                '''

                x_ui = self.urm[u, :].multiply(self.slim_matrix[i]).sum(axis=1)
                x_uj = self.urm[u, :].multiply(self.slim_matrix[j]).sum(axis=1)

                # The difference is computed on the user_seen items
                x_uij = x_ui - x_uj

                # TODO add verbose mode every tot iterations
                # Get current loss
                #loss = self.loss(x_ui, x_uj, x_uij, m)
                #print(f"Current loss is {loss}")

                # Compute gradient of 1/m * log(sigmoid(x_uij))
                gradient = np.sum(1 / (1 + np.exp(x_uij))) / m

                # Update only items corresponding to users in this batch
                items_to_update = self.urm[u, :].sum(axis=0).A > 0

                # Compute overall gradient considering also regularization part
                x_ui_grad = - gradient + self.lambda_i * (x_ui.sum() / m)
                x_uj_grad = + gradient + self.lambda_j * (x_uj.sum() / m)

                # Compute gradients and update parameters
                # Update x_ui parameter
                self.slim_matrix[i] -= lr * x_ui_grad * items_to_update
                self.slim_matrix[i, i] = 0

                # Update x_uj parameter
                self.slim_matrix[j] -= lr * x_uj_grad * items_to_update
                self.slim_matrix[j, j] = 0
                print("TIme " + str(time.time() - t))

    def loss(self, x_ui, x_uj, x_uij, m):
        loss = - (1 / m) * np.sum(np.log(1 / (1 + np.exp(-x_uij))))
        reg = (0.5 / m) * (self.lambda_i * (LA.norm(x_ui) ** 2) + self.lambda_j * (LA.norm(x_uj) ** 2))

        return loss + reg
