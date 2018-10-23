"""
This file contains various utility functions used in most of the recommender algorithms.
"""

import numpy as np


def cosine_similarity(input, alpha=0.5, asym=True, h=0., knn=np.inf, qfunc=None, dtype=np.float32):
    """
    Calculate the cosine similarity

    Parameters
    -------------
    input : sparse matrix
        input matrix (columns represents items for which we want to calculate the similarity coefficients)
    alpha : scalar, optional
        determines type of norm (1 -> norm-1, 0.5 -> norm-2)
    asym : bool, optional
        if True generated matrix is not symmetric (i.e. order matters)
    h : scalar, optional
        shrink term
    knn : integer, optional
        number of nearest neighbours to consider (default: all)
    dtype : data-type, optional
        underlying type on which to operate
    """

    # Compute similarity matrix
    s = (input.T * input).tocsr()

    # Calc norm factors
    norms = input.sum(axis=0).A.ravel()
    if asym:
        assert 0. <= alpha <= 1., "alpha should be a number between 0 and 1"
        norm_factors = np.outer(
            np.power(norms, alpha, dtype=dtype),
            np.power(norms, 1 - alpha, dtype=dtype)
        ) + h

    else:
        norms = np.power(norms, alpha, dtype=dtype)
        norm_factors = np.outer(norms, norms) + h

    # Calculate inverse and normalize
    norm_factors = np.divide(1, norm_factors, out=norm_factors, where=norm_factors != 0, dtype=dtype)
    s = s.multiply(norm_factors).tocsr()
    del norms
    del norm_factors

    # Finally apply qfunc on the individual weights
    if qfunc:
        qfunc = np.vectorize(qfunc)
        s.data = qfunc(s.data)

    # KNN
    if knn != np.inf:
        # For each row
        for row in range(len(s.indptr) - 1):
            # Row offsets
            row_start = s.indptr[row]
            row_end = s.indptr[row + 1]

            # Get row data slice
            row_data = s.data[row_start:row_end]

            if len(row_data) > knn:
                # Discard not meaningful data
                # We take the smallest similarities in the data array
                # and set those data values to 0 using row_start as offset
                # The result is not an actual sparse matrix but it's insanely fast
                discard = np.argpartition(row_data, -knn)[:-knn] + row_start
                s.data[discard] = 0

    # Return computed similarity matrix
    return s


def predict(ratings, targets=None, k=10, mask=None, invert_mask=False):
    """
    Given a (user x item) matrix of ratings, calculate k top items per user

    Parameters
    ---------------
    ratings : sparse matrix
        (user x items) ratings sparse matrix
    targets : list, optional
        list of target users for which we want to predict
    k : integer, optional
        number of items to predict
    mask : sparse matrix, optional
        mask to apply to the ratings matrix to ignore certain items
    invert_mask : bool, optional
        if True, invert the mask (slower)
    """

    # Convert to csr for fast row access
    mask = mask.tocsr()
    ratings = ratings.tocsr()

    # Apply mask
    if mask is not None and not invert_mask:
        ratings = ratings.multiply(mask).tocsr()

    # Compute individually for each user
    preds = []
    for ui in targets:
        # Get row
        ratings_i = ratings.getrow(ui).A.ravel()

        if mask is not None and invert_mask:
            # Apply inverted mask
            ratings_i = ratings_i * (1 - mask.getrow(ui).A.ravel())

        # Compute top k items
        # Using argpartition the complexity is linear
        # in the number of items if k << number of items
        # ------------------------------------
        # Complexity: O(len(items) + k log(k))
        top_idxs = np.argpartition(ratings_i, -k)[-k:]
        sort_idxs = np.argsort(-ratings_i[top_idxs])

        # Add to list
        preds.append((ui, list(top_idxs[sort_idxs])))

    # Return predictions
    return preds
