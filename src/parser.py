import numpy as np
import scipy.sparse as sp

from src.const import *


def parse_tracks(filename="tracks.csv"):
    """
    Builds the tracks matrix #tracks x #attributes (20635 x 4)
    where attributes are track_id,album_id,artist_id,duration_sec
    """
    with open(os.path.join(data_path, filename), "r") as f:
        # Discard first line
        lines = f.readlines()[1:]
        num_lines = len(lines)

        # Sanity check
        assert num_lines == NUM_TRACKS

        # Build matrices
        album_set = sp.dok_matrix((NUM_ALBUMS, NUM_TRACKS), dtype=np.uint8)
        artist_set = sp.dok_matrix((NUM_ARTISTS, NUM_TRACKS), dtype=np.uint8)

        for i, line in enumerate(lines):
            # Parse album and artist
            track, album, artist, _ = [np.int32(i) for i in line.split(",")]
            album_set[album, track] = 1
            artist_set[artist, track] = 1

            print("\rParsing tracks: {:.4}%".format((i / num_lines) * 100), end="")

        print("\n")

        return album_set, artist_set


def parse_interactions(filename="train.csv"):
    """ Parse the train data and return the interaction matrix alone """

    with open(os.path.join(data_path, filename), "r") as f:
        # Discard first line
        lines = f.readlines()[1:]
        num_lines = len(lines)

        # Create container
        interactions = sp.dok_matrix((NUM_PLAYLIST, NUM_TRACKS), dtype=np.uint8)

        for i, line in enumerate(lines):
            playlist, track = [int(i) for i in line.split(",")]
            interactions[playlist, track] = 1

            print("\rParsing interactions: {:.4}%".format((i / num_lines) * 100), end="")

        print("\n")
        # Return matrix
        return interactions


def parse_targets(filename="target_playlists.csv"):
    """
    Builds a list with all the target playlists
    """

    with open(os.path.join(data_path, filename), "r") as f:
        # Discard first line
        lines = f.readlines()[1:]

        targets = []
        for line in lines:
            playlist_id = int(line)
            targets.append(playlist_id)

        return targets
