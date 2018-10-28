import pprint
from random import shuffle

import numpy as np
import os
from timeit import default_timer as timer

from src.const import tune_path
import itertools


class Callable:
    """A callable model. Either a function or a class.

    Attributes
    ----------
    obj : function or class
        Callable model.
    args : list
        Callable's arguments.
    kwargs : dict
        Callable's named arguments.
    """
    def __init__(self, obj, args=None, kwargs=None):
        self.obj = obj
        self.args = args
        self.kwargs = kwargs


class Hyperparameter:
    """Hyperparameter to be tuned.

    Attributes
    ----------
    name : str
        Identifier used to represent the hyperparameter.
    interval : list
        Hyperparameter search space.
    """
    def __init__(self, name, interval):
        self.name = name
        self.interval = interval
        self.value = None

    def random_sample(self):
        """
        Randomly sample and set a value from the search space.
        """
        self.value = np.random.choice(self.interval)


class HyperparameterTuner:
    """Tuner that evaluates a model in a given hyperparameter space.

    Attributes
    ----------
    callable_model : Callable
        Callable of the model to be tuned.
    cartesian_product : bool
        If True search will be done by considering each possible combination of hyperparameters values.
        Otherwise search is performed by sampling randomly in hyperparameters search space.
    cartesian_products : list
        All the possible combinations of hyperparameter values when considering cartesian products.
    max_resample : int
        Maximum number of resampling performed in a sampling iteration from the search space when not
        considering cartesian products.
    resample_try : int
        Number of resample done in a sampling iteration from the search space when not considering cartesian
        products.
    file_name : str
        Name of the file where the results will be written.
    hyperparameters : list
        Hyperparameters to be tuned.
    parse_hyperparam : bool
        True if the hyperparameters have been already extracted from the callable, False otherwise.
    history : dict
        Container to store the evaluation results.

    """
    def __init__(self, callable_model, cartesian_product=False, write_to_file=True, max_resample=1000):
        """Main constructor.

        Parameters
        ---------------
        callable_model : function or class
            Callable of the model to be tuned.
        cartesian_product : bool
            Whether to consider the cartesian products of the hyperparameter values or not.
        write_to_file : bool
            Whether to write on file or not the tuning results.
        max_resample : integer
            Maximum number of resampling performed in a sampling iteration from the search space when not
            considering cartesian products.
        """
        self.callable_model = callable_model
        self.cartesian_product = cartesian_product
        self.max_resample = max_resample

        self.hyperparameters = []

        self.parse_hyperparam = True
        self.parse_callable(callable_model)
        self.parse_hyperparam = False

        assert len(self.hyperparameters) > 0, "You must specify the hyperparameters to search for"

        self.cartesian_products = None
        if self.cartesian_product:

            if len(self.hyperparameters) == 1:
                print("Cartesian product search needs atleast 2 hyperparameters!")
                return

            self.cartesian_products = list(itertools.product(*[h.interval for h in self.hyperparameters]))
            shuffle(self.cartesian_products)

        else:
            self.resample_try = 0

        # Create file to write if write_to_file is set
        self.file_name = None
        if write_to_file:
            self.create_history_file()

        # Dict to store the results obtained for different combination for hyperparameters
        # Key -> Tuple(h1, h2, ..., hn) where hi is the ith hyperparameter value -> MAP@K
        self.history = dict()


    @staticmethod
    def is_callable(entity):
        return isinstance(entity, Callable)

    @staticmethod
    def is_sequence(obj):
        return isinstance(obj, list) or isinstance(obj, tuple) or isinstance(obj, dict)

    def parse_hyperparameter(self, el):
        """
        Parses an element which is not a callable nor a seq to check if it is a hyperparamter.

        Parameters
        ---------------
        el : object (not Callable nor seq)
            The callable to be parsed.

        Returns
        -------
        object (not Callable nor seq9
            New parsed element.
        """
        if isinstance(el, Hyperparameter):
            if self.parse_hyperparam:
                # We are parsing hyperparameter for the first time
                self.hyperparameters.append(el)
            else:
                # We are parsing to substitute hyperparameter with the new value
                return el.value
        else:
            return el

    def parse_callable(self, callable_thing):
        """
        Parses a callable for hyperparameters.

        Parameters
        ---------------
        callable_model : Callable
            The callable to be parsed.

        Returns
        -------
        Callable
            New parsed callable.
        """
        args = []
        if callable_thing.args:
            for arg in callable_thing.args:

                if self.is_callable(arg):
                    args.append(self.parse_callable(arg))
                if self.is_sequence(arg):
                    args.append(self.parse_sequence(arg))
                else:
                    args.append(self.parse_hyperparameter(arg))

        kwargs = dict()
        if callable_thing.kwargs:
            for k, v in callable_thing.kwargs.items():

                if self.is_callable(v):
                    kwargs.update({k: self.parse_callable(v)})
                elif self.is_sequence(v):
                    kwargs.update({k: self.parse_sequence(v)})
                else:
                    kwargs.update({k: self.parse_hyperparameter(v)})

        return callable_thing.obj(*args, **kwargs)

    def parse_sequence(self, seq):
        """
        Parses a sequence for hyperparameters.

        Parameters
        ---------------
        seq : list or tuple or dict
            The sequence ot be parsed.

        Returns
        -------
        seq
            New parsed sequence.
        """
        if isinstance(seq, list) or isinstance(seq, tuple):

            # Treat both list and tuples as lists
            new_seq = []
            for i, el in enumerate(seq):
                if self.is_callable(el):
                    new_seq.append(self.parse_callable(el))
                elif self.is_sequence(el):

                    new_seq.append(self.parse_sequence(el))
                else:
                    new_seq.append(self.parse_hyperparameter(el))

            return new_seq
        else:  # it is a dict
            new_d = dict()

            for k, v in seq.items():
                if self.is_callable(v):
                    new_d.update({k: self.parse_callable(v)})
                elif self.is_sequence(v):
                    new_d.update({k: self.parse_callable(v)})
                else:
                    new_d.update({k: self.parse_hyperparameter(v)})

            return new_d

    def new_random_samples(self):
        """
        Update and return new values for the hyperparameters by random sampling from their intervals.

        Returns
        -------
        tuple
            New hyperparameters values.
        """
        for h in self.hyperparameters:
            h.random_sample()
        return tuple(h.value for h in self.hyperparameters)

    def new_product_samples(self, product):
        """
        Update and return new values for the hyperparameters using a given product.

        Parameters
        ---------------
        product : tuple
            Values for each hyperparameter.

        Returns
        -------
        tuple
            New hyperparameters values.
        """
        for i, h_value in enumerate(product):
            self.hyperparameters[i].value = h_value
        return tuple(h.value for h in self.hyperparameters)

    def create_history_file(self):
        os.makedirs(tune_path, exist_ok=True)
        print("Preparing file where tuning results will be written ...")
        h_names = ",".join(h.name for h in self.hyperparameters)
        self.file_name = h_names + "@pid" + str(os.getpid()) + ".csv"

        with open(tune_path + "/" + self.file_name, "w") as f:
            f.write(h_names + "," + "MAP@10\n")

    def write_history(self):
        if self.file_name:
            print("Writing to file results ...")
            with open(tune_path + "/" + self.file_name, "a") as f:
                for k, v in self.history.items():
                    s_map = "{:.5f}".format(v)
                    f.write(",".join(str(el) for el in k) + "," + s_map)
                    f.write("\n")

    def print_history(self):
        pprint.pprint(self.history, width=1)

    def run(self):
        try:
            while True:
                if self.cartesian_product:

                    # We checked all the possible values
                    if not self.cartesian_products:
                        print("Exhausted all possible combinations, printing results ...")
                        self.print_history()
                        self.write_history()
                        return

                    # Sample new hyperparameter values from their products
                    new_product = self.cartesian_products.pop()
                    new_hyperparameters = self.new_product_samples(new_product)

                else:
                    # Randomly sample from the hyperparameters intervals
                    # If we can't find a new combination, just terminate
                    new_hyperparameters = self.new_random_samples()
                    while new_hyperparameters in self.history:
                        # Try to find a combination not already evaluated
                        self.resample_try += 1

                        if self.resample_try == self.max_resample:
                            # We probably exhausted all the combinations in the search space
                            print("Interrupted tuning because of max resample, printing results ...")
                            self.print_history()
                            self.write_history()
                            return

                        new_hyperparameters = self.new_random_samples()

                    self.resample_try = 0

                # Run evaluation and store the results in the history
                start = timer()
                print("Evaluating model with the following parameters")
                o_string = "\n".join(h.name + ": " + str(h.value) for h in self.hyperparameters)
                print(o_string)
                map_at_k = self.parse_callable(self.callable_model).evaluate()
                print("Evaluated model with the following parameters")
                print(o_string)
                print("Got MAP@10: {:.5f}\n".format(map_at_k))
                print("elapsed: {:.3f}s\n".format(timer() - start))
                self.history.update({new_hyperparameters: map_at_k})

        except KeyboardInterrupt:
            print("Interrupted tuning, printing results ...")
            self.print_history()
            self.write_history()

