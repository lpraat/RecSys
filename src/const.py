import os

# Constants
NUM_TRACK_ATTRIBUTES = 4
NUM_PLAYLIST = 50446
NUM_TRACKS = 20635
NUM_ALBUMS = 12744
NUM_ARTISTS = 6668
NUM_TARGETS = 10000
NUM_RECOMMENDATIONS_PER_PLAYLIST = 10
NUM_INTERACTIONS = 1211791

# Paths
data_path = os.path.dirname(os.path.realpath(__file__)) + "/../data"
cache_path = os.path.dirname(os.path.realpath(__file__)) + "/../cache"
