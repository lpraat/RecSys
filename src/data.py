import os
import pickle

from src.parser import parse_train_set

data_path = os.path.dirname(os.path.realpath(__file__)) + "/../data"
cache_path = os.path.dirname(os.path.realpath(__file__)) + "/../cache"


class Cache:


    def __init__(self):
        """
        Create cache and load all data files
        """
        # Load data
        self.assets = {}
        self.load_all()
        pass


    def load_all(self):
        """
        Load all data files
        """

        # Get train data
        interactions    = self.load_file("interactions.obj")
        train_set       = self.load_file("train_set.obj")
        test_set        = self.load_file("test_set.obj")
        if interactions == None or train_set == None or test_set == None:
            # Parse train and test sets
            interactions, train_set, test_set = parse_train_set("train.csv")
            self.save_file("interactions.obj", interactions)
            self.save_file("train_set.obj", train_set)
            self.save_file("test_set.obj", test_set)
        
        # Load in cache
        self.set_records([
            ("interactions", interactions),
            ("train_set", train_set),
            ("test_set", test_set)
        ])


    def get_record(self, file):
        """
        Fetch record from cache
        """

        # Return cached result or none
        if file in self.assets:
            return self.assets[file]
        else:
            return None


    def set_record(self, file, obj):
        """
        Set a cache record
        """

        # Save object in cache
        self.assets.update({file:obj})

    
    def set_records(self, records):
        """
        Set multiple records at once
        """

        for record in records:
            self.set_record(record[0], record[1])
    

    def load_file(self, file):
        """
        Tries to load a file from persistent cache
        """

        try:
            f = open(cache_path + "/" + file, "rb")
            return pickle.load(f)
        
        except FileNotFoundError:
            return None


    def save_file(self, file, obj):
        """
        Save an object to the persistent cache and return the object
        """

        if obj != None:
            # Create directory if necessary
            os.makedirs(cache_path, exist_ok=True)

            with open(cache_path + "/" + file, "wb") as f:
                pickle.dump(obj, f)
        
        # Return object
        return obj

