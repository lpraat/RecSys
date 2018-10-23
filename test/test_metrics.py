import unittest

from src.metrics import ap_at_k, map_at_k, leave_one_out


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

    def test_leave_one_out(self):
        preds = [
            [1, 2, 3, 4],
            [4, 5, 2, 6],
            [8, 1, 23, 5, 78],
            [0, 23, 4, 7]
        ]
        test_set = [3, 7, 1, 4]

        self.assertEqual(leave_one_out(preds, test_set), 0.75)
