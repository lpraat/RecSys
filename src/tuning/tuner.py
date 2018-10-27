from random import shuffle

import numpy as np
from timeit import default_timer as timer
import itertools
import pprint
import os

# TODO add some docs
from src.const import tune_path


class Callable:
    def __init__(self, obj, args=None, kwargs=None):
        self.obj = obj
        self.args = args
        self.kwargs = kwargs


class Hyperparameter:
    def __init__(self, name, interval):
        self.name = name
        self.interval = interval
        self.value = None

    def random_sample(self):
        self.value = np.random.choice(self.interval)


def hyperparameter_search(callable_model, hyperparameters, cartesian_product=False, write_to_file=True,
                          max_resample=1000):
    def is_callable(entity):
        return isinstance(entity, Callable)

    def is_sequence(obj):
        return isinstance(obj, list) or isinstance(obj, tuple) or isinstance(obj, dict)

    def check_hyperparameter(v):
        if isinstance(v, Hyperparameter):
            return v.value
        else:
            return v

    def parse_sequence(seq):
        if isinstance(seq, list) or isinstance(seq, tuple):

            # Treat both list and tuples as lists
            new_seq = []
            for i, el in enumerate(seq):
                if is_callable(el):
                    new_seq.append(parse_callable(el))
                elif is_sequence(el):

                    new_seq.append(parse_sequence(el))
                else:
                    new_seq.append(check_hyperparameter(el))

            return new_seq
        else:  # it is a dict
            new_d = dict()

            for k, v in seq.items():
                if is_callable(v):
                    new_d.update({k: parse_callable(v)})
                elif is_sequence(v):
                    new_d.update({k: parse_callable(v)})
                else:
                    new_d.update({k: check_hyperparameter(v)})

            return new_d

    def parse_callable(callable_thing):

        args = []
        if callable_thing.args:
            for arg in callable_thing.args:

                if is_callable(arg):
                    args.append(parse_callable(arg))
                if is_sequence(arg):
                    args.append(parse_sequence(arg))
                else:
                    args.append(check_hyperparameter(arg))

        kwargs = dict()
        if callable_thing.kwargs:
            for k, v in callable_thing.kwargs.items():

                if is_callable(v):
                    kwargs.update({k: parse_callable(v)})
                elif is_sequence(v):
                    kwargs.update({k: parse_sequence(v)})
                else:
                    kwargs.update({k: check_hyperparameter(v)})

        return callable_thing.obj(*args, **kwargs)

    def new_random_samples():
        for h in hyperparameters:
            h.random_sample()
        return tuple(h.value for h in hyperparameters)

    def new_product_samples(product):
        for i, h_value in enumerate(product):
            hyperparameters[i].value = h_value
        return tuple(h.value for h in hyperparameters)

    def write_history(filename, history):
        if filename:
            print("Writing to file results ...")
            with open(tune_path + "/" + filename, "a") as f:
                for k, v in history.items():
                    s_map = "{:.5f}".format(v)
                    f.write(",".join(str(el) for el in k) + "," + s_map)
                    f.write("\n")

    assert len(hyperparameters) > 0, "You must specify the hyperparameters to search for"

    cartesian_products = None
    if cartesian_product:

        if len(hyperparameters) == 1:
            print("Cartesian product search needs atleast 2 hyperparameters!")
            return

        cartesian_products = list(itertools.product(*[h.interval for h in hyperparameters]))
        shuffle(cartesian_products)

    # Create file to write if write_to_file is set
    file_name = None
    if write_to_file:
        print("Preparing file where tuning results will be written ...")
        h_names = ",".join(h.name for h in hyperparameters)
        file_name = h_names + "@pid" + str(os.getpid()) + ".csv"

        with open(tune_path + "/" + file_name, "w") as f:
            f.write(h_names + "," + "MAP@10\n")

    # Dict to store the results obtained for different combination for hyperparameters
    # Key -> Tuple(h1, h2, ..., hn) where hi is the ith hyperparameter value -> MAP@K
    history = dict()
    resample_try = 0

    try:
        while True:
            # Sample new hyperparameters from their intervals for a new run
            if cartesian_product:

                # We checked all the possible values
                if not cartesian_products:
                    print("Exhausted all possible combinations, printing results ...")
                    pprint.pprint(history, width=1)
                    write_history(file_name, history)
                    return

                new_product = cartesian_products.pop()
                new_hyperparameters = new_product_samples(new_product)

            else:
                # Randomly sample from the hyperparameters intervals
                # If we can't find a new combination, just terminate
                new_hyperparameters = new_random_samples()
                while new_hyperparameters in history:
                    # Try to find a combination not already evaluated
                    resample_try += 1

                    if resample_try == max_resample:
                        # We probably exhausted all the combinations in the search space
                        print("Interrupted tuning because of max resample, printing results ...")
                        pprint.pprint(history, width=1)
                        write_history(file_name, history)
                        return

                    new_hyperparameters = new_random_samples()

                resample_try = 0

            # Run evaluation and store the results in the history
            start = timer()
            print("Evaluating model with the following parameters")
            o_string = "\n".join(h.name + ": " + str(h.value) for h in hyperparameters)
            print(o_string)
            map_at_k = parse_callable(callable_model).evaluate()
            print("Evaluated model with the following parameters")
            print(o_string)
            print("Got MAP@10: {:.5f}\n".format(map_at_k))
            print("elapsed: {:.3f}s\n".format(timer() - start))
            history.update({new_hyperparameters: map_at_k})

    except KeyboardInterrupt:
        print("Interrupted tuning, printing results ...")
        pprint.pprint(history, width=1)
        write_history(file_name, history)
