"""
This file contains the base class for all recommender algorithms.
"""

from timeit import default_timer as timer

from src.data import Cache
from src.metrics import evaluate
from .utils import predict


class RecSys:
    """
    Abstract base class every recommender is built upon.

    Attributes
    ----------
    dataset : str
        Name of the dataset from cache to be used to generate recommendations.
    cache : Cache
        The global cache where all the data is kept.
    """

    def __init__(self):
        """ Default constructor """

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


    def run(self, dataset=None, targets=None, k=10):
        """
        Computes k predictions for each target

        Parameters
        -----------
        dataset : sparse matrix or string
            Input (user x items) interactions matrix to use
            If a sparse matrix is provided it should be in csr format
            If a string is passed, it searches for it in the global cache
        targets : list
            An ordered list of users for which to compute the predictions
            If no list is provided, predictions are computed for all users
            of the input interactions matrix
        k : scalar
            Number of items to recommend (default: 10)
        """

        # Get dataset
        if dataset is not None:
            dataset = self.cache.fetch(dataset) if isinstance(dataset, str) else dataset
        else:
            dataset = self.cache.fetch("interactions")
        dataset = dataset.tocsr()

        # Targets
        if targets is not None:
            targets = self.cache.fetch(targets) if isinstance(targets, str) else targets
        else:
            targets = range(dataset.shape[0])

        # Compute ratings
        ratings = self.rate(dataset)

        print("predicting ...")
        start = timer()
        # Predict
        preds = predict(ratings, targets=targets, k=k, mask=dataset, invert_mask=True)
        print("elapsed: {:.3f}s\n".format(timer() - start))
        del ratings

        return preds


    def evaluate(self, train_set="train_set", test_set="test_set", targets=None, k=10):
        """
        Evaluate system using a train set and a test set

        Right now this method uses MAP@k metric to evaluate the performance

        Parameters
        -----------
        train_set : sparse matrix or string
            Usually a subset of the interactions matrix used to compute k predictions
            If a string is passed, the train set is fetched from the cache
            By default, the method expects to find a 'train_set' record in the cache
        test_set : list of lists
            List of target predictions, one for each user in the interactions matrix
            If no value is provided the method searches for a 'test_set' record in the cache
        k : scalar
            Number of predictions on which to evaluate
        """

        test_set = self.cache.fetch(test_set) if isinstance(test_set, str) else test_set

        # Predict
        preds = self.run(dataset=train_set, targets=targets, k=k)

        print("evaluating model ...")
        # Evaluate model
        score = evaluate(preds, test_set)
        print("MAP@{}: {:.5f}\n".format(k, score))

        return score

    def compute_similarity(self, dataset):
        """
        Computes similarity matrix
        """
        raise NotImplementedError


    def rate(self, dataset):
        """
        Computes items ratings for target users

        Return a matrix

        dataset : sparse matrix or string
            Input (user x items) interactions matrix to use
            If a sparse matrix is provided it should be in csr format
            If a string is passed, it searches for it in the global cache
        targets : list
            An ordered list of users for which to compute the ratings
            If no list is provided, predictions are computed for all users
            of the input interactions matrix
        """

        raise NotImplementedError
