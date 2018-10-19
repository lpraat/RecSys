"""
Perform an item-based collaborative filtering algorithm
to determine the ranking of each item for each user
"""

import numpy as np
import scipy.sparse as sp


def get_rankings(interactions, *contexts, weights = [1], normalize = [True]):
    """ Given one ore more interaction matrices generate a ranking matrix from the similarity of the items """

    # Compute the similarity matrix
    print("computing similarity matrix ...\n")

    similarity = interactions.transpose().tocsr().dot(interactions.tocsc()) * (weights[0] / (interactions.shape[1] if normalize[0] else 1))
    for i, context in enumerate(contexts):
        similarity += context.transpose().tocsr().dot(context.tocsc()) * (weights[i + 1] / (context.shape[1] if normalize[i + 1] else 1))
    
    # Compute the ranking matrix
    print("computing ranking matrix ...\n")

    rankings = interactions.tocsr().dot(similarity.tocsc())

    return rankings