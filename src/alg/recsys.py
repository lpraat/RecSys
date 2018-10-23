"""
This file contains the base class for all recommender algorithms.
"""

from src.data import Cache


class RecSys:
    """Abstract base class every recommender is built upon.

    Attributes
    ----------
    dataset : str
        Name of the dataset from cache to be used to generate recommendations.
    cache : Cache
        The global cache where all the data is kept.
    """

    def __init__(self, dataset="train_set"):

        # Global cache
        global cache

        # Create cache if necessary
        # This ensures that only one global cache exists
        # Don't reuse memory!
        try:
            self.cache = cache
        except NameError:
            cache = Cache()
            self.cache = cache

        # Initial dataset
        self.dataset = dataset

    def run(self, targets=None, k=10):
        """ Get k predictions for each target user """
        raise NotImplementedError('every recommender must override run')

    def evaluate(self, train_set="train_set", test_set="test_set", k=10):
        """ Evaluate model on train set using MAP@k metric """
        raise NotImplementedError('every recommender must override evaluate')
