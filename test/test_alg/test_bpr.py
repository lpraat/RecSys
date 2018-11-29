import unittest

import scipy.sparse as sp

from src.alg.slim import BPRSampler


class TestBPRSampler(unittest.TestCase):
    def test_sample(self):
        urm = sp.csr_matrix([
            [0, 0, 1, 0, 1, 1]*100,
            [0, 1, 0, 1, 1, 0]*100,
            [0, 0, 1, 1, 0, 1]*100
        ])

        bpr_sampler = BPRSampler(urm)

        for _ in range(1000):
            sample = bpr_sampler.sample()
            u, i, j = sample
            self.assertTrue(i != j)
            self.assertTrue(urm[u, i] == 1, urm[u, j] == 0)

        bpr_sampler = BPRSampler(urm.T)

        for _ in range(1000):
            sample = bpr_sampler.sample()
            u, i, j = sample
            print(u, i, j)
            self.assertTrue(i != j)
            self.assertTrue(urm[i, u] == 1, urm[j, u] == 0)


