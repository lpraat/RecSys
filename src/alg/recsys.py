"""
This file contains the base class for all recommender algorithms
"""

from src.data import Cache


class RecSys:
    """ Base class for recommender systems """
    

    def __init__(self):
        """ Base constructor """

        # Create cache
        self.cache = Cache()
    

    def run(self, targets):
        raise NotImplementedError