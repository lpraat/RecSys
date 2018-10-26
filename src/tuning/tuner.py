import numpy as np
from timeit import default_timer as timer
import itertools
import pprint


# TODO add some docs and tests

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


# TODO Wrap this in function which automatically creates a pool to run multiple evaluation in parallel
def hyperparameter_search(callable_model, hyperparameters, cartesian_product=False):
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

    def get_new_samples():
        for h in hyperparameters:
            h.random_sample()
        return (h.value for h in hyperparameters)

    # Dict to store the results obtained for different combination for hyperparameters
    # Key -> Tuple(h1, h2, ..., hn) where hi is the ith hyperparameter value -> MAP@K
    history = dict()

    try:
        while True:
            # Sample new hyperparameters from their intervals for a new run

            if cartesian_product:
                # Sample from the cartesian products
                cartesian_products = list(itertools.product(h.intervals for h in hyperparameters))
                new_hyperparameters = cartesian_products.pop()

                # We checked all the possible values
                if not cartesian_products:
                    # Print the history on multiple lines, where each line is a key, value entry
                    print("Exhausted all possible combinations, printing results ...")
                    pprint.pprint(history, width=1)

            else:
                # Randomly sample from the hyperparameters intervals
                new_hyperparameters = get_new_samples()
                while new_hyperparameters in history:
                    new_hyperparameters = get_new_samples()

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
