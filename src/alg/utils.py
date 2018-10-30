"""
This file contains various utility functions used in most of the recommender algorithms
"""

import numpy as np
import scipy.sparse as sp
import random
from timeit import default_timer as timer


def cosine_similarity(input, alpha=0.5, asym=True, h=0., knn=np.inf, qfunc=None, dtype = np.float32):
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
    
    # Calc norm factors
    norms = input.sum(axis=0).A.ravel().astype(dtype)

    if (not asym or alpha == 0.5) and h == 0:
        # If matrix is symmetric and h is zero
        # this optimization makes it a lot faster
        # and much more memory-friendly
        norm_input = input * sp.diags(np.divide(
            1,
            np.power(norms, alpha, out=norms, dtype=dtype),
            out=norms,
            where=norms != 0,
            dtype=dtype
        ), format="csr", dtype=dtype)

        # Compute similarity matrix
        s = (norm_input.T * norm_input)
    
    else:
        # Compute similarity matrix
        s = (input.T * input).tocsr()

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
            
        # Recompute sparsity
        s = recompute_sparsity(s)

    # Finally apply qfunc on the individual weights
    if qfunc:
        qfunc = np.vectorize(qfunc)
        s.data = qfunc(s.data)
    
    # Return computed similarity matrix
    return s


def clusterize(input, s=None, k=8):
    """
    Given a (user x item) matrix, divide users in clusters based on similarity between users

    Parameters
    ---------------
    input : sparse csr matrix
        (user x item) interactions matrix
    s : sparse csr matrix
        pre calculated similarity matrix
    k : integer
        number of clusters
    """

    # Require a csr matrix for fast row access
    assert isinstance(input, sp.csr_matrix), "csr_matrix required, {} given".format(type(input))
    
    if s is not None:
        # Sanity check
        assert s.shape[0] == s.shape[1] and s.shape[0] == input.shape[0], "similarity matrix dimensions don't match"
        assert isinstance(s, sp.csr_matrix), "csr_matrix required, {} given".format(type(s))
    else:
        print("computing cosine similarities between users ...")
        start = timer()
        # Compute similarity between users
        s = cosine_similarity(input.T, dtype=np.float32)
        print("elapsed time: {:.3f}s".format(timer() - start))
    
    print("computing clusters of similar users ...")
    start = timer()
    # Randomly pick center for first cluster
    u0 = np.random.randint(input.shape[0])

    # Choose k - 1 furthest points from u0
    clusters = [[u0]]
    similarities = s.getrow(u0).A.ravel()
    for i in range(1, k):
        # Take furthest point
        ui = np.argmin(similarities)
        clusters.append([ui])

        # Add similarities
        similarities += s.getrow(ui).A.ravel()

    # Compute clusters centers for convenience
    cluster_centers = [cl[0] for cl in clusters]

    # Row indices pointers of subsets, we start computing this here
    subsets_indptr = [[
        0,
        input.indptr[cl[0] + 1] - input.indptr[cl[0]]
    ] for cl in clusters]
    # An incremental offset
    subsets_indptr_off = [np.max(subsets_indptr[ki]) for ki, _ in enumerate(clusters)]

    # Place each user in appropriate cluster
    for ui in [i for i in range(input.shape[0]) if i not in cluster_centers]:
        # Get row
        row_i = s.getrow(ui).A.ravel()

        # Compute similarities w.r.t each cluster
        sim = []
        for cl in clusters:
            # Original averages on all sample
            sim.append(np.average(row_i[cl]))
        
        # Put in cluster which nearest point is the absolute nearest
        ki = np.argmax(sim)
        clusters[ki].append(ui)
        
        # Determine
        num_indices = input.indptr[ui + 1] - input.indptr[ui]
        subsets_indptr[ki].append(num_indices + subsets_indptr_off[ki])
        subsets_indptr_off[ki] += num_indices
    
    print("elapsed time: {:.3f}s".format(timer() - start))

    print("splitting matrix in clusters ...")
    start = timer()
    # Create cluster matrices
    counters = [0 for _ in clusters]
    subsets = [sp.csr_matrix((
        # Data
        np.ones(subsets_indptr_off[ki], dtype=np.uint8),
        # Indices
        np.zeros(subsets_indptr_off[ki], dtype=np.uint16),
        # Indptr
        subsets_indptr[ki]
    ), shape=(len(clusters[ki]), input.shape[1])) for ki in range(len(clusters))]
    for ki, cl in enumerate(clusters):
        for ui in cl:
            # Get row ptrs
            input_row_start = input.indptr[ui]
            input_row_end = input.indptr[ui + 1]
            subset_row_start = subsets[ki].indptr[counters[ki]]
            subset_row_end = subsets[ki].indptr[counters[ki] + 1]
            counters[ki] += 1

            # Copy row slice
            subsets[ki].indices[subset_row_start:subset_row_end] = input.indices[input_row_start:input_row_end]

    print("elapsed time: {:.3f}s".format(timer() - start))
    
    # Return clusters
    return clusters, subsets


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
    
    # Compute individually for each user
    preds = []
    for ui in targets:
        # Get row
        ratings_i_start = ratings.indptr[ui]
        ratings_i_end = ratings.indptr[ui + 1]
        ratings_i_data = ratings.data[ratings_i_start:ratings_i_end]
        ratings_i_indices = ratings.indices[ratings_i_start:ratings_i_end]

        if mask is not None:
            # Apply inverted mask
            mask_i_start = mask.indptr[ui]
            mask_i_end = mask.indptr[ui + 1]
            mask_i_indices = mask.indices[mask_i_start:mask_i_end]

            masked_i = np.in1d(ratings_i_indices, mask_i_indices, assume_unique=True, invert=invert_mask)
            ratings_i_data = ratings_i_data[masked_i]
            ratings_i_indices = ratings_i_indices[masked_i]

        # Compute top k items
        # Using argpartition the complexity is linear
        # in the number of non-zero items if k << number of nnz
        # ------------------------------------
        # Complexity: O(len(nnz) + k log(k))
        if len(ratings_i_data) > k:
            top_idxs = np.argpartition(ratings_i_data, -k)[-k:]
            ratings_i_indices = ratings_i_indices[top_idxs]
            sort_idxs = np.argsort(-ratings_i_data[top_idxs])
        else:
            ratings_i_data.resize(k)
            ratings_i_indices.resize(k)
            sort_idxs = np.argsort(-ratings_i_data)

        # Add to list
        preds.append((ui, list(ratings_i_indices[sort_idxs])))

    # Return predictions
    return preds


def recompute_sparsity(mat):
    """
    Recompute the sparsity of a sparse matrix (i.e. remove zero elements)
    Future operations can benefit from this

    Parameters:
    mat : sparse matrix
        input matrix to recompute
    """

    # Use a csr matrix
    if not isinstance(mat, sp.csr_matrix):
        mat = mat.tocsr()

    # First extract indices of non-zero elements
    data_i = mat.data.nonzero()[0]
    data = mat.data[data_i]

    # Extract column indices of non-zero elements
    indices = mat.indices[data_i]

    # Update indptr offsets
    indptr = [0]
    for row_i in range(mat.shape[0]):
        # Row start and end
        row_start = mat.indptr[row_i]
        row_end = mat.indptr[row_i + 1]

        # Push number of non-zero elements
        num_nonzero = np.count_nonzero(mat.data[row_start:row_end])
        indptr.append(indptr[row_i] + num_nonzero)
    
    # Rebuild matrix
    return sp.csr_matrix((
        data,
        indices,
        indptr
    ), shape=mat.shape)