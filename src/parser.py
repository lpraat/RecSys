import numpy as np
import os

NUM_TRACK_ATTRIBUTES = 4
NUM_PLAYLIST = 50446
NUM_TRACKS = 20635
NUM_TARGETS = 10000
NUM_RECOMMENDATIONS_PER_SONG = 10

data_path = os.path.dirname(os.path.realpath(__file__)) + "/../data"


def parse_tracks():
    """
    Builds the tracks matrix #tracks x #attributes (20635 x 4)
    where attributes are track_id,album_id,artist_id,duration_sec
    """
    with open(data_path + '/tracks.csv', 'r') as f:
        lines = f.readlines()[1:]
        num_tracks = len(lines)
        tracks_matrix = np.zeros((num_tracks, NUM_TRACK_ATTRIBUTES), dtype=np.int32)

        for index, line in enumerate(lines):
            track_id, album_id, artist_id, duration_sec = [int(i) for i in line.split(",")]
            tracks_matrix[index] = np.array([track_id, album_id, artist_id, duration_sec])

    return tracks_matrix


def parse_interactions():
    """
    Builds the interactions (sparse) matrix #playlist x #items (50446 x 20635)
    If playlist i has item(track) j then interactions_matrix[i][j] = 1 otherwise 0
    """
    with open(data_path + '/train.csv', 'r') as f:
        lines = f.readlines()[1:]

        num_playlist = NUM_PLAYLIST
        num_tracks = NUM_TRACKS

        interactions_matrix = np.zeros((num_playlist, num_tracks), dtype=np.int32)

        for line in lines:
            playlist_id, track_id = [int(i) for i in line.split(",")]
            interactions_matrix[playlist_id][track_id] = 1

    return interactions_matrix


def parse_targets():
    """
    Builds the matrix to be filled with recommendations.
    It is a matrix #targets x #recommendations+1 (10000 x 11).

    Why 11 columns?
    Column 0 is the id of the target playlist
    Column 1 to 10 are the recommendations for that playlist
    """

    with open(data_path + '/target_playlists.csv', 'r') as f:
        lines = f.readlines()[1:]

        targets_matrix = np.zeros((NUM_TARGETS, NUM_RECOMMENDATIONS_PER_SONG + 1), dtype=np.int32)

        for index, line in enumerate(lines):
            playlist_id = int(line)
            targets_matrix[index][0] = playlist_id

    return targets_matrix

# To visualize data
# print(parse_tracks())
# print(parse_interactions())
# print(parse_targets())
