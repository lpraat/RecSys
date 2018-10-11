import unittest

from src.metrics import ap_at_k, map_at_k


class TestMetrics(unittest.TestCase):

    def test_map(self):
        preds = [
            [6, 4, 7, 1, 2],
            [6, 4, 7, 1, 2]
        ]

        targets = [
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5]
        ]

        self.assertEqual(ap_at_k(preds[0], targets[0], k=2), 0.25)
        self.assertEqual(map_at_k(preds, targets, k=2), 0.25)


