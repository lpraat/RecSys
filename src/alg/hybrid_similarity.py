from src.alg.recsys import RecSys


class HybridSimilarity(RecSys):

    def __init__(self, *models):

        self.models = list(models)



