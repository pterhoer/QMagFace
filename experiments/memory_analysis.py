import tracemalloc
from datasets.paired import PairDataset
from datasets.single import SingleDataset
from similarity.cosine import Cosine
from similarity.qmagface import QMagFace


def main(db, dataset_root):
    tracemalloc.start()
    pds = PairDataset(SingleDataset(dataset_root, db), None)
    cur, peak0 = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    tracemalloc.start()

    cos_scores = Cosine.similarity(pds.embeddings, pds.pairs)
    cur, peak1 = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    tracemalloc.start()

    qmf = QMagFace()
    qmf.train(pds.embeddings, pds.pairs, pds.matches)
    cur, peak2 = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    tracemalloc.start()

    qmf.similarity(pds.embeddings, pds.pairs)
    cur, peak3 = tracemalloc.get_traced_memory()

    print(f'Database: {db}')
    print(f'Number of Pairs: {len(pds.pairs)}')
    print(f'Dataset initialization peak Memory consumption: {peak0 / 1e6} MB')
    print(f'Cosine Similarity peak Memory consumption: {peak1 / 1e6} MB')
    print(f'Training peak Memory consumption: {peak2 / 1e6} MB')
    print(f'QMagFace evaluation peak Memory consumption: {peak3 / 1e6} MB')