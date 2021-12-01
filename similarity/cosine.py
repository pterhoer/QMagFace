import sklearn
import numpy as np
from similarity.base import Similarity


class Cosine(Similarity):
    @staticmethod
    def similarity(f1, f2, is_normed=False):
        result = np.zeros((len(f1), ))
        if not is_normed:
            f1_normed = sklearn.preprocessing.normalize(f1)
            f2_normed = sklearn.preprocessing.normalize(f2)
        else:
            f1_normed = f1
            f2_normed = f2
        if len(f1.shape) < 2:
            f1_normed = [f1_normed]
            f2_normed = [f2_normed]
        for i, (f1_, f2_) in enumerate(zip(f1_normed, f2_normed)):
            result[i] = np.sum(f1_ * f2_)
        return result

    def name(self):
        return "Cosine"
