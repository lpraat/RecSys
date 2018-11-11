import unittest
import scipy.sparse as sp

from src.alg.slim import BPRSampler


class TestBPRSampler(unittest.TestCase):
    def test_sample(self):
        urm = sp.csr_matrix([
            [0, 1, 1, 1, 1, 1]*8000,
            [0, 1, 1, 1, 1, 1]*8000,
            [0, 1, 1, 1, 1, 1]*8000
        ])

        bpr_sampler = BPRSampler(urm)

        for i in range(1000):
            sample = bpr_sampler.sample()
            u, i, j = sample
            self.assertTrue(i != j)
            self.assertTrue(urm[u, i] == 1, urm[u, j] == 0)
