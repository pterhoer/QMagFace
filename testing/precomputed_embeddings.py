import numpy as np
from utils.pairs_files import convert_pairs
from evaluation.metrics import evaluate_metrics
from similarity.cosine import Cosine
from similarity.qmagface import QMagFace


def test_cplfw():
    pairs_path = '../_data/pairs/lfw_pairs.txt'
    embeddings_path = '../_data/lfw_comp/lfw_emb.npy'
    filenames_path = '../_data/lfw_comp/lfw_fns.npy'
    qmagface = QMagFace(0.077428, 0.125926)
    filenames = np.load(filenames_path)
    filenames = np.array(["./lfw/" + fn.split('/')[-1] for fn in filenames])
    embeddings = np.load(embeddings_path)
    fns, pairs, matches = convert_pairs(pairs_path, 'lfw')
    f1 = []
    f2 = []
    for pa, pb in pairs:
        try:
            idxa = np.where(pa == filenames)[0][0]
            idxb = np.where(pb == filenames)[0][0]
            f1.append(embeddings[idxa])
            f2.append(embeddings[idxb])
        except:
            pass
    f1 = np.array(f1)
    f2 = np.array(f2)
    cosine_scores = Cosine.similarity(f1, f2)
    qmagface_scores = qmagface.similarity(f1, f2)
    metrics = ['acc']
    cosine_acc = evaluate_metrics(cosine_scores, matches, metrics)['acc']
    qmagface_acc = evaluate_metrics(qmagface_scores, matches, metrics)['acc']
    print(cosine_acc)
    print(qmagface_acc)

test_cplfw()