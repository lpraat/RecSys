import os
import numpy as np
import scipy.sparse as sp

from src.const import *

def parse_tracks():
    """
    Builds the tracks matrix #tracks x #attributes (20635 x 4)
    where attributes are track_id,album_id,artist_id,duration_sec
    """
    with open(data_path + '/tracks.csv', 'r') as f:
        lines = f.readlines()[1:]
        num_tracks = len(lines)
        tracks_matrix = np.zeros((num_tracks, NUM_TRACK_ATTRIBUTES), dtype = np.int32)

        for index, line in enumerate(lines):
            track_id, album_id, artist_id, duration_sec = [int(i) for i in line.split(",")]
            tracks_matrix[index] = np.array([track_id, album_id, artist_id, duration_sec])

    return tracks_matrix


def parse_interactions(filename = "train.csv"):
    """ Parse the train data and return the interaction matrix alone """

    with open(os.path.join(data_path, filename), "r") as f:
        # Discard first line
        lines = f.readlines()[1:]
        num_lines = float(len(lines))

        # Create container
        interactions = sp.dok_matrix((NUM_PLAYLIST, NUM_TRACKS), dtype=np.int32)

        for i, line in enumerate(lines):
            playlist, track = [int(i) for i in line.split(",")]
            interactions[playlist, track] = 1

            # Debug
            print("\033[2J")
            print("parsing interactions: {:.2}".format(i / num_lines))

        # Return matrix
        return interactions


def parse_targets():
    """
    Builds the matrix to be filled with recommendations
    It is a matrix #targets x #recommendations+1 (10000 x 11)

    Why 11 columns?
    Column 0 is the id of the target playlist
    Column 1 to 10 are the recommendations for that playlist
    """

    with open(data_path + '/target_playlists.csv', 'r') as f:
        lines = f.readlines()[1:]

        targets_matrix = np.zeros((NUM_TARGETS, NUM_RECOMMENDATIONS_PER_PLAYLIST + 1), dtype=np.int32)

        for index, line in enumerate(lines):
            playlist_id = int(line)
            targets_matrix[index][0] = playlist_id

    return targets_matrix

# To visualize data
#print(parse_tracks())
#print(parse_interactions_old())
#print(parse_targets())
#print(load_interactions("/train.csv"))