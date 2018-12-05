import unittest

from src.alg.smart_ensemble.smart_ensemble import borda_count


class TestUtils(unittest.TestCase):

    def test_borda_count(self):
        # example from https://it.wikipedia.org/wiki/Metodo_Borda
        A = [[1, 2, 3, 4], 0.42]
        B = [[2, 3, 4, 1], 0.26]
        C = [[3, 4, 2, 1], 0.15]
        D = [[4, 3, 2, 1], 0.17]

        self.assertEqual(borda_count([A, B, C, D], n=4), [2, 3, 1, 4])

