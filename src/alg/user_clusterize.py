"""
This file contains a recommender system that first splits the dataset
in k clusters of most similar users, and then runs the requested
sub recsys on each of these clusters
"""

import multiprocessing as mp
from timeit import default_timer as timer

from .recsys import RecSys
from .utils import clusterize


class UserClusterize(RecSys):


    def __init__(self, model, k=4):
        """
        Constructor

        Parameters
        -----------
        model : recsys
            Model to apply to generated clusters
        k : integer
            Number of clusters to generate
        """

        # Parent constructor
        super().__init__()

        # initial values
        self.model = model
        self.k = k

    
    def run(self, dataset=None, targets=None, k=10):

        # Get dataset
        if dataset is not None:
            dataset = self.cache.fetch(dataset) if isinstance(dataset, str) else dataset
        else:
            dataset = self.cache.fetch("interactions")
        dataset = dataset.tocsr()

        # Targets
        if targets is not None:
            targets = self.cache.fetch(targets) if isinstance(targets, str) else targets
        else:
            targets = range(dataset.shape[0])

        # Compute clusters
        clusters, subsets = clusterize(dataset, k=self.k)

        # Run model on clusters
        global run_model
        def run_model(subset):
            return self.model.run(subset, k=k)

        # Use a pool of k threads (max. 16)
        workers = mp.Pool(min(self.k, 16))
        preds = workers.map(run_model, subsets)
        workers.close()

        print("computing final predictions ...")
        start = timer()
        # Sort predictions
        preds_final = [(ui, []) for ui in range(dataset.shape[0])]
        for ki, pred in enumerate(preds):
            for uj, items in pred:
                # Set prediction for user
                ui = clusters[ki][uj]
                preds_final[ui][1].extend(items)
        print("elapsed: {:.3f}s\n".format(timer() - start))

        # Return target slice
        return [preds_final[ui] for ui in targets]