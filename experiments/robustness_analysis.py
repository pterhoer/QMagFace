import pandas as pd
from datasets.paired import PairDataset
from similarity.cosine import Cosine
from similarity.qmagface import QMagFace
from evaluation.metrics import evaluate_metrics


def main(result_path, paths, dbs):
    params = [(0.092861, 0.135311), (0.065984, 0.103799), (0.077428, 0.125926)]

    columns = ['db', 'model', 'eer', 'fmr1e-1', 'fmr1e-2', 'fmr1e-3', 'fmr1e-4', 'fmr1e-5', 'auc']
    metrics = columns[2:]
    df = pd.DataFrame(columns=columns)
    for root, (alpha, beta) in zip(paths, params):
        qmf = QMagFace(alpha, beta)
        base_model = root.split('/')[-1]
        clfs = [qmf, Cosine()]
        models = [f'q{base_model}', base_model]
        for db in dbs:
            pds = PairDataset((root, db))
            for clf, model in zip(clfs, models):
                scores = clf.similarity(pds.embeddings, pds.pairs)
                stats = evaluate_metrics(scores, pds.matches, metrics, cross_validation=False)
                stats['model'] = model
                stats['db'] = db
                df = pd.concat([df, pd.DataFrame({k: [v] for k, v in stats.items()})])
    print(df)
    df.to_csv(result_path, sep='\t')
