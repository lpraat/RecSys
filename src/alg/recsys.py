"""
This file contains the base class for all recommender algorithms
"""

from src.data import Cache


class RecSys:
    """ Base class for recommender systems """
    

    def __init__(self, dataset = "train_set"):
        """ Base constructor """

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
        raise NotImplementedError

    
    def evaluate(self, train_set="train_set", test_set="test_set", k=10):
        raise NotImplementedError