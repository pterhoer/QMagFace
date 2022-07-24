import os
import numpy as np
import pickle
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import timeit
import sklearn
import argparse
import warnings
import tqdm
from menpo.visualize.viewmatplotlib import sample_colours_from_colourmap
from prettytable import PrettyTable
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing
from pathlib import Path
from itertools import product


def main(source_root, dest_root, target):
    matplotlib.use('Agg')

    warnings.filterwarnings("ignore")
    model = 'magface_r100'

    recalculate = True
    m = 0.125926
    b = 0.077428

    def combine_mean(fs):
        return np.mean(fs, 0)

    def combine_weighted_f_q(fs):
        qs = np.array([np.linalg.norm(f) for f in fs])
        f = np.sum([f * qs[i] for i, f in enumerate(fs)], 0)
        q = np.sum(qs ** 2) / np.sum(qs)
        return f / np.linalg.norm(f) * q

    def read_template_media_list(path):
        ijb_meta = pd.read_csv(path, sep=' ', header=None).values
        templates = ijb_meta[:, 1].astype(np.int)
        medias = ijb_meta[:, 2].astype(np.int)
        return templates, medias

    def read_template_pair_list(path):
        pairs = pd.read_csv(path, sep=' ', header=None).values
        t1 = pairs[:, 0].astype(np.int)
        t2 = pairs[:, 1].astype(np.int)
        label = pairs[:, 2].astype(np.int)
        return t1, t2, label

    def get_image_feature(img_path, img_list_path, filenames, embeddings):
        img_list = open(img_list_path)
        files = img_list.readlines()
        print('files:', len(files))
        faceness_scores = []
        img_feats = []
        for img_index, each_line in enumerate(files):
            name_lmk_score = each_line.strip().split(' ')
            faceness_score = name_lmk_score[-1]
            img_name = name_lmk_score[0]
            feat_indx = np.searchsorted(filenames, img_name)
            if img_name != filenames[feat_indx]:
                continue
            img_feats.append(embeddings[feat_indx])
            faceness_scores.append(faceness_score)
        img_feats = np.array(img_feats).astype(np.float32)
        faceness_scores = np.array(faceness_scores).astype(np.float32)

        # img_feats = np.ones( (len(files), 1024), dtype=np.float32) * 0.01
        # faceness_scores = np.ones( (len(files), ), dtype=np.float32 )
        return img_feats, faceness_scores

    def image2template_feature(img_feats=None, templates=None, medias=None):
        # ==========================================================
        # 1. face image feature l2 normalization. img_feats:[number_image x feats_dim]
        # 2. compute media feature.
        # 3. compute template feature.
        # ==========================================================
        unique_templates = np.unique(templates)
        template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))

        for count_template, uqt in enumerate(unique_templates):
            (ind_t,) = np.where(templates == uqt)
            face_norm_feats = img_feats[ind_t]
            face_medias = medias[ind_t]
            unique_medias, unique_media_counts = np.unique(face_medias,
                                                           return_counts=True)
            media_norm_feats = []
            for u, ct in zip(unique_medias, unique_media_counts):
                (ind_m,) = np.where(face_medias == u)
                if ct == 1:
                    media_norm_feats += [face_norm_feats[ind_m]]
                else:  # image features from the same video will be aggregated into one feature
                    media_norm_feats += [
                        combine_func(face_norm_feats[ind_m])
                    ]
            media_norm_feats = np.array(media_norm_feats)
            template_feats[count_template] = combine_func(media_norm_feats)
        return template_feats, unique_templates

    def verification(template_norm_feats=None,
                     unique_templates=None,
                     p1=None,
                     p2=None,
                     d=0):
        # ==========================================================
        #         Compute set-to-set Similarity Score.
        # ==========================================================
        template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
        for count_template, uqt in enumerate(unique_templates):
            template2id[uqt] = count_template

        score = np.zeros((len(p1),))  # save cosine distance between pairs

        total_pairs = np.array(range(len(p1)))
        batchsize = 100000  # small batchsize instead of all pairs in one batch due to the memory limiation
        sublists = [
            total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)
        ]
        total_sublists = len(sublists)
        for c, s in tqdm.tqdm(enumerate(sublists), total=total_sublists):
            feat1 = np.squeeze(template_norm_feats[template2id[p1[s]]], 1)
            feat2 = np.squeeze(template_norm_feats[template2id[p2[s]]], 1)
            norm_feat1 = sklearn.preprocessing.normalize(feat1)
            norm_feat2 = sklearn.preprocessing.normalize(feat2)
            q1 = feat1[:, 0] / norm_feat1[:, 0]
            q2 = feat2[:, 0] / norm_feat2[:, 0]
            q = np.min(np.stack([q1, q2], 1), 1)
            similarity_score = np.sum(norm_feat1 * norm_feat2, -1)
            if use_qa_score:
                cutoff_value = 0
                weight = (m * similarity_score - b)
                weight[weight > cutoff_value] = 0
                score[s] = weight * q + similarity_score
            else:
                score[s] = similarity_score
        return score

    def read_score(path):
        with open(path, 'rb') as fid:
            img_feats = pickle.load(fid)
        return img_feats

    def common_str_gen(combine_func, use_qa_score, target):
        return f'{dest_root}/{combine_func.__name__}_{["cos", "qa"][use_qa_score]}_{target.lower()}'

    save_path = dest_root
    targets = [target]
    tests = [(combine_mean, False), (combine_weighted_f_q, True)]
    # tests += tests_extra
    for target, (combine_func, use_qa_score) in product(targets, tests):
        print(target, combine_func.__name__, use_qa_score)
        if not recalculate:
            break

        assert target == 'IJBC' or target == 'IJBB'

        # =============================================================
        # load image and template relationships for template feature embedding
        # tid --> template id,  mid --> media id
        # format:
        #           image_name tid mid
        # =============================================================
        start = timeit.default_timer()
        templates, medias = read_template_media_list(
            os.path.join(source_root, '%s_face_tid_mid.txt' % target.lower()))
        stop = timeit.default_timer()
        print('Time: %.2f s. ' % (stop - start))

        # =============================================================
        # load template pairs for template-to-template verification
        # tid : template id,  label : 1/0
        # format:
        #           tid_1 tid_2 label
        # =============================================================
        start = timeit.default_timer()
        p1, p2, label = read_template_pair_list(f'{source_root}{target.lower()}_template_pair_label.txt')
        stop = timeit.default_timer()
        print('Time: %.2f s. ' % (stop - start))

        # # Step 2: Get Image Features
        # =============================================================
        # load image features
        # format:
        #           img_feats: [image_num x feats_dim] (227630, 512)
        # =============================================================
        start = timeit.default_timer()
        img_path = './%s/loose_crop' % target
        img_list_path = f'{source_root}{target.lower()}_name_5pts_score.txt'
        filenames = np.load(f'{source_root}filenames_{target.lower()}_{model}.npy')
        embeddings = np.load(f'{source_root}emb_{target.lower()}_{model}.npy')
        args = np.argsort(filenames)
        filenames = filenames[args]
        filenames = np.array([filename.split('\\')[-1] for filename in filenames])
        embeddings = embeddings[args]
        img_feats, faceness_scores = get_image_feature(img_path, img_list_path, filenames, embeddings)
        stop = timeit.default_timer()
        print('Time: %.2f s. ' % (stop - start))
        print('Feature Shape: ({} , {}) .'.format(img_feats.shape[0],
                                                  img_feats.shape[1]))

        # # Step3: Get Template Features
        # =============================================================
        # compute template features from image features.
        # =============================================================
        start = timeit.default_timer()
        # ==========================================================
        # Norm feature before aggregation into template feature?
        # Feature norm from embedding network and faceness score are able to decrease weights for noise samples (not face).
        # ==========================================================
        # 1. FaceScore （Feature Norm）
        # 2. FaceScore （Detector）

        img_input_feats = img_feats
        template_norm_feats, unique_templates = image2template_feature(
            img_input_feats, templates, medias)
        stop = timeit.default_timer()
        print('Time: %.2f s. ' % (stop - start))

        # # Step 4: Get Template Similarity Scores

        # =============================================================
        # compute verification scores between template pairs.
        # =============================================================
        start = timeit.default_timer()
        score = verification(template_norm_feats, unique_templates, p1, p2)
        stop = timeit.default_timer()
        print('Time: %.2f s. ' % (stop - start))

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        common_str = common_str_gen(combine_func, use_qa_score, target)
        score_save_file = common_str + '.npy'
        quality_save_file = common_str + '_quality.npy'
        label_save_file = common_str + '_label.npy'
        np.save(score_save_file, score)
        np.save(label_save_file, label)

    # # Step 5: Get ROC Curves and TPR@FPR Table
    files = []
    scores_save_folder = f''
    for target, (combine_func, use_qa_score) in product(targets, tests):
        common_str = common_str_gen(combine_func, use_qa_score, target)
        score_save_file = common_str + '.npy'
        label_save_file = common_str + '_label.npy'
        files.append((score_save_file, label_save_file))
    methods = []
    scores = []
    qualities = []
    labels = []
    for score_file, label_file in files:
        methods.append(Path(score_file).stem)
        scores.append(np.load(score_file))
        labels.append(np.load(label_file))

    methods = np.array(methods)
    scores = dict(zip(methods, scores))
    labels = dict(zip(methods, labels))
    colours = dict(
        zip(methods, sample_colours_from_colourmap(methods.shape[0], 'Set2')))
    # x_labels = [1/(10**x) for x in np.linspace(6, 0, 6)]
    x_labels = [10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]
    tpr_fpr_table = PrettyTable(['Methods'] + [str(x) for x in x_labels])
    fig = plt.figure()
    for method in methods:
        fpr, tpr, _ = roc_curve(labels[method], scores[method])
        roc_auc = auc(fpr, tpr)
        fpr = np.flipud(fpr)
        tpr = np.flipud(tpr)  # select largest tpr at same fpr
        np.save(f'{dest_root}/roc_{method}.npy', np.array([fpr, tpr]))
        plt.plot(fpr,
                 tpr,
                 lw=1,
                 label=('[%s (AUC = %0.4f %%)]' %
                        (method.split('-')[-1], roc_auc * 100)))
        tpr_fpr_row = []
        tpr_fpr_row.append(f'{method}'.replace('combine_', '').replace('_', ' '))
        for fpr_iter in np.arange(len(x_labels)):
            _, min_index = min(
                list(zip(abs(fpr - x_labels[fpr_iter]), range(len(fpr)))))
            mindex = np.argmin(np.abs(fpr - x_labels[fpr_iter]))
            # tpr_fpr_row.append('%.4f' % tpr[min_index])
            tpr_fpr_row.append('%.2f' % (tpr[min_index] * 100))

        tpr_fpr_table.add_row(tpr_fpr_row)
        x = scores[method]

    plt.xlim([10 ** -6, 0.1])
    plt.ylim([0.3, 1.0])
    plt.grid(linestyle='--', linewidth=1)
    plt.xticks(x_labels)
    plt.yticks(np.linspace(0.3, 1.0, 8, endpoint=True))
    plt.xscale('log')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC on IJB')
    plt.legend(loc="lower right")
    # plt.show()
    fig.savefig(os.path.join(save_path, '%s.pdf' % target.lower()))
    fig.savefig(os.path.join(save_path, '%s.png' % target.lower()))
    print(tpr_fpr_table)
