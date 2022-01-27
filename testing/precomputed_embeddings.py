import numpy as np
from evaluation.metrics import evaluate_metrics
from similarity.cosine import Cosine
from similarity.qmagface import QMagFace


def test_cplfw():
    pairs_path = '_data/pairs/cplfw_pairs.txt'
    embeddings_path = '_data/magface100/embeddings_cplfw.npy'
    filenames_path = '_data/magface100/filenames_cplfw.npy'
    qmagface = QMagFace(0.077428, 0.125926)
    filenames = np.load(filenames_path)
    embeddings = np.load(embeddings_path)
    with open(pairs_path) as file:
        lines = file.readlines()
    f1 = []
    f2 = []
    matches = []
    for line in lines:
        a, b, m = map(int, line.replace('.jpg', '').split(' '))
        filename_a = f'./cplfw/{int(a - 1)}.jpg'
        filename_b = f'./cplfw/{int(b - 1)}.jpg'
        idx_a = np.where(filename_a == filenames)[0][0]
        idx_b = np.where(filename_b == filenames)[0][0]
        f1.append(embeddings[idx_a])
        f2.append(embeddings[idx_b])
        matches.append(int(m > 0))
    matches = np.array(matches)
    f1 = np.array(f1)
    f2 = np.array(f2)
    cosine_scores = Cosine.similarity(f1, f2)
    qmagface_scores = qmagface.similarity(f1, f2)
    metrics = ['acc']
    cosine_acc = evaluate_metrics(cosine_scores, matches, metrics)['acc']
    qmagface_acc = evaluate_metrics(qmagface_scores, matches, metrics)['acc']
    assert np.isclose(cosine_acc, 93.00)
    assert np.isclose(qmagface_acc, 93.50)
