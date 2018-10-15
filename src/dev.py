# process here train data to create both training and test set
# put here evaluation
import random
import pickle
import numpy as np
import scipy.sparse as sp

from src.const import *

def build_test_set(interaction_matrix):
    
    # Create test set
    test_set = []

    for i in range(0, NUM_PLAYLIST):
        keys    = list(interaction_matrix[i].keys())
        idx     = random.randint(0, len(keys) - 1)

        # Add to test set
        test_set.append(keys[idx][1])

        # Completion string
        print(chr(27) + "[2J")
        print(("completion: {0:.2f}" + "%").format(i * 100 / float(NUM_PLAYLIST)))

    return test_set