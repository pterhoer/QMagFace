import os
import click
import tqdm
from similarity.cosine import Cosine
from similarity.qmagface import QMagFace
from datasets.paired import PairDataset
from evaluation.metrics import evaluate_metrics


def main(train_db, alpha, beta, dataset_root, pairs_root, test_db):
    qmf = QMagFace()
    if alpha is not None and beta is not None:
        qmf = QMagFace(alpha, beta)
    elif train_db is not None:
        train_pds = PairDataset((dataset_root, train_db))
        qmf.train(train_pds.embeddings, train_pds.pairs, train_pds.matches)
        print(f'Found parameters alpha = {qmf.alpha:.6f} beta = {qmf.beta:.6f}')
    else:
        raise Exception('QMagFace cannot function without training or precomputed parameters.')

    qmf_accs = []
    cos_accs = []
    for db in tqdm.tqdm(test_db):
        pairs_file = os.path.join(pairs_root, f'{db}_pairs.txt')
        pds = PairDataset((dataset_root, db), pairs_file)
        qmf_scores = qmf.similarity(pds.embeddings, pds.pairs)
        cos_scores = Cosine.similarity(pds.embeddings, pds.pairs)
        qmf_metrics = evaluate_metrics(qmf_scores, pds.matches, ['acc'])
        cos_metrics = evaluate_metrics(cos_scores, pds.matches, ['acc'])
        qmf_accs.append(qmf_metrics['acc'])
        cos_accs.append(cos_metrics['acc'])

    print('db', 'QMagFace', 'Cosine', sep='\t')
    for db, qmf_acc, cos_acc in zip(test_db, qmf_accs, cos_accs):
        print(f'{db}\t{qmf_acc:.2f}\t{cos_acc:.2f}')