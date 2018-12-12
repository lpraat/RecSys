from lightfm import LightFM

from src.alg.recsys import RecSys
import scipy.sparse as sp


class Light(RecSys):

    def __init__(self):
        super().__init__()

    def compute_similarity(self, dataset):
        raise NotImplementedError

    def rate(self, dataset):
        # Instantiate and train the model
        model = LightFM(loss='warp', no_components=200)
        model.fit(dataset, epochs=50, num_threads=2, verbose=True)

        user_embeddings = sp.csr_matrix(model.user_embeddings)
        item_embeddings = sp.csr_matrix(model.item_embeddings)

        return user_embeddings * item_embeddings.T