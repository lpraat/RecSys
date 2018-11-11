import unittest
import scipy.sparse as sp
import numpy as np

from src.alg.bpr import BPRSampler
from src.alg.slim import Slim


class TestSlim(unittest.TestCase):

    def test_gradients(self):

        def gradient_checking(batch_size, lambda_i, lambda_j):
            urm = sp.csr_matrix([
                [0, 1, 1, 0, 0, 0],
                [0, 1, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, 1]
            ], dtype=np.int16)

            assert batch_size <= urm.nnz, "Invalid batch size"
            lr = 0.01
            np.random.seed(1)
            slim = Slim(lambda_i=lambda_i, lambda_j=lambda_j)
            slim.urm = urm
            slim.num_interactions = urm.nnz
            slim.num_users = urm.shape[0]
            slim.num_items = urm.shape[1]
            slim.slim_matrix = np.random.randn(slim.num_items, slim.num_items)
            slim.bpr_sampler = BPRSampler(urm)
            slim_matrix_test = np.copy(slim.slim_matrix)

            np.testing.assert_array_almost_equal(slim_matrix_test, slim.slim_matrix)

            np.random.seed(1)
            batches = slim.build_batches(batch_size)
            np.random.seed(1)

            for batch in batches:
                u = batch[:, 0]
                i = batch[:, 1]
                j = batch[:, 2]

                x_ui = urm[u, :].multiply(slim_matrix_test[i]).sum(axis=1)
                x_uj = urm[u, :].multiply(slim_matrix_test[j]).sum(axis=1)
                x_uij = x_ui - x_uj

                gradient = np.sum(1 / (1 + np.exp(x_uij))) / batch.shape[0]
                items_to_update = urm[u, :].sum(axis=0).A > 0
                x_ui_grad = - gradient + lambda_i * (x_ui.sum() / batch.shape[0])
                x_uj_grad = + gradient + lambda_j * (x_uj.sum() / batch.shape[0])

                slim_matrix_test[i] -= lr * x_ui_grad * items_to_update
                slim_matrix_test[i, i] = 0

                slim_matrix_test[j] -= lr * x_uj_grad * items_to_update
                slim_matrix_test[j, j] = 0

            slim.train(lr=lr, batch_size=batch_size, num_epochs=1)
            np.testing.assert_array_almost_equal(slim_matrix_test, slim.slim_matrix)

            def grad_check(param, grad):
                eps = 1e-7
                i_param = x_ui + eps if param == "i" else x_ui
                j_param = x_uj + eps if param == "j" else x_uj
                x_uij_plus = i_param - j_param
                loss_plus = slim.loss(i_param, j_param, x_uij_plus, batch_size)

                i_param = x_ui - eps if param == "i" else x_ui
                j_param = x_uj - eps if param == "j" else x_uj
                x_uij_minus = i_param - j_param
                loss_minus = slim.loss(i_param, j_param, x_uij_minus, batch_size)
                grad_approx = (loss_plus - loss_minus) / (2 * eps)

                return np.linalg.norm(grad_approx - grad) / (np.linalg.norm(grad) + np.linalg.norm(grad_approx))

            self.assertTrue(grad_check(param="i", grad=x_ui_grad) < 2e-7)
            self.assertTrue(grad_check(param="j", grad=x_uj_grad) < 2e-7)

        # Perform some gradient checking both when using regularization and when not
        gradient_checking(batch_size=1, lambda_i=0, lambda_j=0)
        gradient_checking(batch_size=6, lambda_i=0, lambda_j=0)
        gradient_checking(batch_size=3, lambda_i=0.5, lambda_j=0)
        gradient_checking(batch_size=1, lambda_i=0.0001, lambda_j=0.4)

