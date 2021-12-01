import numpy as np
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold


def roc_curve_kfold(matches, scores, nrof_folds=10):
    thr = np.linspace(np.min(scores), np.max(scores), 600)
    nrof_pairs = min(len(matches), scores.shape[0])
    nrof_thresholds = len(thr)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds, ))

    indices = np.arange(nrof_pairs)
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds, ))
        for threshold_idx, threshold in enumerate(thr):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, scores[train_set], matches[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thr):
            fprs[fold_idx, threshold_idx], tprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
                                                                                                 scores[test_set],
                                                                                                 matches[
                                                                                                     test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thr[best_threshold_index], scores[test_set],
                                                      matches[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return fpr, tpr, thr, accuracy


def calculate_accuracy(threshold, scores, matches):
    predict_issame = scores > threshold
    tp = np.sum(np.logical_and(predict_issame, matches))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(matches)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(matches)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), matches))
    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / scores.size
    return fpr, tpr, acc


def evaluate_metrics(scores, matches, metrics, cross_validation=True, return_roc=False):
    stats = {}
    acc = 0
    if cross_validation:
        fpr, tpr, thr, acc = roc_curve_kfold(matches, scores)
    else:
        fpr, tpr, thr = roc_curve(matches, scores)
        if 'acc' in metrics:
            acc = max([calculate_accuracy(t, scores, matches)[2] for t in thr])
    fnr = 1 - tpr
    for metric in metrics:
        if metric == 'eer':
            stats[metric] = fpr[np.argmin(np.absolute((fnr - fpr)))] * 100
        elif metric == 'threer':
            stats[metric] = thr[np.argmin(np.absolute((fnr - fpr)))]
        elif 'fmr' in metric:
            exp = float(metric[3:])
            stats[metric] = np.min(1 - tpr[fpr <= exp]) * 100
        elif 'thr' in metric:
            exp = float(metric[3:])
            stats[metric] = thr[np.argmin(1 - tpr[fpr <= exp])]
        elif metric == 'auc':
            stats[metric] = auc(fpr, tpr) * 100
        elif metric == 'acc':
            stats['acc'] = np.mean(acc) * 100
        elif metric == 'acc_std':
            stats[metric] = np.std(acc * 100)
    if return_roc:
        return stats, (tpr, fpr, thr)
    return stats
