import tqdm
from time import time
from datasets.paired import PairDataset
from datasets.single import SingleDataset
from similarity.cosine import Cosine
from similarity.qmagface import QMagFace


def main(train_db, dataset_root):
    runs = 10
    db = train_db
    pds = PairDataset(SingleDataset(dataset_root, db), None)
    print(f'Database: {db}')
    print(f'Runs: {runs}')
    print(f'Number of Pairs: {len(pds.pairs)}')

    train_times = []
    cos_eval_times = []
    qmf_eval_times = []
    for i in tqdm.trange(runs):
        t0 = time()
        qmf = QMagFace()
        qmf.train(pds.embeddings, pds.pairs, pds.matches)
        train_times.append(time() - t0)

        t0 = time()
        cos_scores = Cosine.similarity(pds.embeddings, pds.pairs)
        cos_eval_times.append(time() - t0)

        t0 = time()
        qmf.similarity(pds.embeddings, pds.pairs)
        qmf_eval_times.append(time() - t0)

    print(f'Training took on average: {sum(train_times) / runs:.4f}s')
    print(f'Cosine evaluation took on average: {sum(cos_eval_times) / runs:.4f}s')
    print(f'QMagFace evaluation took on average: {sum(qmf_eval_times) / runs:.4f}s')
