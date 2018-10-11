import unittest

from src.parser import parse_tracks, parse_interactions, parse_targets


class TestParser(unittest.TestCase):

    def test_tracks_parser(self):
        tracks_matrix = parse_tracks()

        self.assertEqual(tracks_matrix.shape, (20635, 4))
        self.assertEqual(tracks_matrix[0][3], 167)
        self.assertEqual(tracks_matrix[20634][1], 12529)

    def test_parse_train(self):
        interactions_matrix = parse_interactions()

        self.assertEqual(interactions_matrix.shape, (50446, 20635))
        self.assertEqual(interactions_matrix[0][1220], 1)
        self.assertEqual(interactions_matrix[89][7814], 1)
        self.assertEqual(interactions_matrix[50445][6971], 1)

    def test_parse_targets(self):
        targets_matrix = parse_targets()

        self.assertEqual(targets_matrix.shape, (10000, 11))
        self.assertEqual(targets_matrix[-1][0], 50424)


