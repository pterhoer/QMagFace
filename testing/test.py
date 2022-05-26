from datasets.single import SingleDataset
from datasets.paired import PairDataset
from similarity.cosine import Cosine
from similarity.qmagface import QMagFace
from evaluation.metrics import evaluate_metrics


def main():
    sds = SingleDataset('../_data/magface100/', 'lfw')
    pds = PairDataset(sds, '../_data/pairs/lfw_pairs.txt')
    f = pds.embeddings
    p = pds.pairs
    m = pds.matches
    score = Cosine.similarity(f, p)
    qmf = QMagFace(alpha=0.077428, beta=0.125926)
    qmf_score = qmf.similarity(f, p)
    qmf_metrics = evaluate_metrics(qmf_score, m, ['acc'])
    cos_metrics = evaluate_metrics(score, m, ['acc'])
    print(qmf_metrics['acc'])
    print(cos_metrics['acc'])


if __name__ == '__main__':
    main()
