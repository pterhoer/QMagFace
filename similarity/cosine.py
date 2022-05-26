import sklearn
import sklearn.preprocessing
import numpy as np
from similarity.base import Similarity


class Cosine(Similarity):
    @staticmethod
    def similarity(f, pair_indices, is_normed=False):
        if not is_normed:
            f = sklearn.preprocessing.normalize(f)
        return np.sum(f[pair_indices[:, 0]] * f[pair_indices[:, 1]], axis=1)

    def name(self):
        return "Cosine"
