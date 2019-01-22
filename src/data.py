"""
This file contains functions to build the train set,
load files from cache and store data in a runtime global cache class
"""

import math
import os
import pickle
import random
from timeit import default_timer as timer

import numpy as np

from src.const import cache_path, NUM_PLAYLIST
from src.parser import parse_interactions, parse_tracks, parse_targets


def build_train_set_fixed(interactions, k=1):
    # Output variables
    items = list(interactions.items())
    train_set = interactions.copy()
    test_set = []

    pos = 0
    for playlist_id in range(NUM_PLAYLIST):
        # Get tracks
        tracks = []
        for i in range(pos, len(items)):
            key = items[i][0]

            if playlist_id != key[0]:
                break

            # Add to track list
            tracks.append(key[1])
            pos += 1

        # Adjust k to have at least one track in playlist
        k = len(tracks) - 1 if len(tracks) <= k else k

        # Generate indices to extract from interactions
        # This indices are added to the test set
        indices = []
        test_set_i = []
        for _ in range(k):
            t = random.randint(0, len(tracks) - 1)
            while t in indices:
                t = random.randint(0, len(tracks) - 1)
            indices.append(t)

            # Remove and add to test set
            track_id = tracks[t]
            train_set[playlist_id, track_id] = 0
            test_set_i.append(track_id)
        test_set.append(test_set_i)

        # Debug
        # print("building train set: {}".format(playlist_id))

    # Return built sets
    return train_set, test_set


def build_train_set_uniform(interactions, targets=None, p=0.15):
    """
    Given the interactions matrix and a list of targets
    creates a train set by extracting a uniform number of
    items per target from the interactions matrix

    Parameters
    ---------------
    interactions : sparse matrix
        (users x items) interactions matrix
    targets : list
        list of targets (if not specified all users)
    p : scalar
        a number between 0. and 1. that specified the amount
        of data to extract
    """

    if targets is None:
        targets = range(interactions.shape[0])

    print("building train set ...")
    start = timer()
    # Output variables
    train_set = interactions.copy()
    test_set = {}

    csr_inter = interactions.tocsr()

    for playlist_id in targets:
        # Get tracks
        tracks = csr_inter.indices[csr_inter.indptr[playlist_id]:csr_inter.indptr[playlist_id + 1]]

        # Calc number of tracks to extract
        k = math.ceil(len(tracks) * p)

        # Generate indices to extract from interactions
        # This indices are added to the test set
        indices = []
        test_set_i = []
        for _ in range(k):
            t = random.randint(0, len(tracks) - 1)
            while t in indices:
                t = random.randint(0, len(tracks) - 1)
            indices.append(t)

            # Remove and add to test set
            track_id = tracks[t]
            train_set[playlist_id, track_id] = 0
            test_set_i.append(track_id)

        test_set[playlist_id] = test_set_i

        # Debug
        # print("building train set: {}".format(playlist_id))
    print("elapsed time: {:.3f}s\n".format(timer() - start))

    # Return built sets
    return train_set, test_set


def load_file(filename):
    """ Tries to load a file from persistent cache """

    try:
        f = open(cache_path + "/" + filename, "rb")
        return pickle.load(f)

    except FileNotFoundError:
        return None


def save_file(filename, obj):
    """ Save an object to the persistent cache and return the object """

    if obj != None:
        # Create directory if necessary
        os.makedirs(cache_path, exist_ok=True)

        with open(cache_path + "/" + filename, "wb") as f:
            pickle.dump(obj, f)

    # Return object
    return obj


class Cache:
    def __init__(self):
        """ Create cache and load required data in cache """
        # Load data
        self.assets = {}
        self.load_all()

    def load_all(self):
        """ Load all data files """

        # Get train data
        interactions = load_file("interactions.obj")
        train_set = load_file("train_set.obj")
        test_set = load_file("test_set.obj")

        album_set = load_file("album_set.obj")
        artist_set = load_file("artist_set.obj")

        targets = load_file("targets.obj")

        if interactions is None:
            interactions = parse_interactions()
            save_file("interactions.obj", interactions)

        if album_set is None or artist_set is None:
            album_set, artist_set = parse_tracks()
            save_file("album_set.obj", album_set)
            save_file("artist_set.obj", artist_set)

        if targets is None:
            targets = parse_targets("target_playlists.csv")
            save_file("targets.obj", targets)

        if train_set is None or test_set is None:
            train_set, test_set = build_train_set_uniform(interactions, targets, 0.2)
            save_file("train_set.obj", train_set)
            save_file("test_set.obj", test_set)

        # Load in cache
        self.store_multi({
            "interactions": interactions,
            "train_set": train_set,
            "test_set": test_set,
            "album_set": album_set,
            "artist_set": artist_set,
            "targets": targets,
            "top_pop": np.array([2272, 18266, 13980, 2674, 17239, 10496, 15578, 5606, 10848, 8956])
        })

    def fetch(self, key):
        """
        Get a cache record associated with the provided key
        """

        # Return cached result or none
        if key in self.assets:
            return self.assets[key]
        else:
            return None

    def store(self, file, obj):
        """
        Set a cache record
        """

        # Save object in cache
        self.assets.update({file: obj})

    def store_multi(self, records):
        """
        Set multiple records at once
        """

        self.assets.update(records)
