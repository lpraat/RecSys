import unittest

from src.data import load_file, save_file
from src.parser import parse_interactions


class TestData(unittest.TestCase):
    def test_build_train_set_uniform(self):
        # Load interaction matrix
        interactions = load_file("interactions.obj").tocsr()
        if interactions is None:
            interactions = parse_interactions().tocsr()
            save_file("interactions.obj", interactions)

        # Build train set
        train_set = load_file("train_set.obj").tocsr()
        test_set = load_file("test_set.obj")
        # train_set, test_set = build_train_set_uniform(interactions, 0.1)
        diff = interactions - train_set

        diff = diff.tocsr()
        for i in range(diff.shape[0]):
            tracks = [key[1] for key in diff.getrow(i).todok().keys()]
            self.assertEqual(sorted(test_set[i]), sorted(tracks))
