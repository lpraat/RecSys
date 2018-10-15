import os
import pickle

from src.parser import parse_interactions
from src.dev import build_test_set

data_path = os.path.dirname(os.path.realpath(__file__)) + "/../data"
cache_path = os.path.dirname(os.path.realpath(__file__)) + "/../cache"


class Cache:


    def __init__(self):
        # Load data
        self.assets = {}
        self.load_all()
        pass


    def load_all(self):
        # Load data in cache
        self.set_record("interactions", self.load_file("train.csv", (parse_interactions, "train.csv")))
        self.set_record("test_set", self.load_file("test.obj", (build_test_set, self.get_record("interactions"))))
        # @todo: create train set and add to cache

    def get_record(self, file):
        # Return cached result or none
        if file in self.assets:
            return self.assets[file]
        else:
            return None


    def set_record(self, file, obj):
        # Save object in cache
        self.assets.update({file:obj})
    

    def load_file(self, file, process_func):
        """
        Loads a file from /cache or parses it from /data and serializes it
        """

        try:
            f = open(cache_path + "/" + os.path.splitext(file)[0] + ".obj", "rb")
            return pickle.load(f)
        
        except FileNotFoundError:
            self.save_file(file, process_func[0](process_func[1]))

    def save_file(self, file, obj):
        # Create directory if necessary
        os.makedirs(cache_path, exist_ok=True)

        with open(cache_path + "/" + os.path.splitext(file)[0] + ".obj", "wb") as f:
            pickle.dump(obj, f)

