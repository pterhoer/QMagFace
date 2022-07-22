import re
import typing

from .single import SingleDataset
import numpy as np


def get_prefix(filenames, from_back):
    return '/'.join(filenames[0].split('/')[:-from_back])


def generate_pairs(labels, n=5, limit=10):
    distinct_labels = np.unique(labels).astype(int)
    genuine_pairs = []
    imposter_pairs = []
    for label in distinct_labels:
        genuine_idxs = np.argwhere(labels == label).squeeze()
        lgi = genuine_idxs.size
        if lgi > limit:
            genuine_idxs = np.random.choice(genuine_idxs, (limit,), replace=False)
            lgi = limit
        imposter_idxs = np.argwhere(labels != label).squeeze()
        for i in range(lgi):
            for j in range(i + 1, lgi):
                genuine_pairs.append([genuine_idxs[i], genuine_idxs[j]])
                imposters = np.random.randint(0, imposter_idxs.shape[0], (2, n))
                for k in range(n):
                    imposter_pairs.append([genuine_idxs[i], imposter_idxs[imposters[0, k]]])
                    imposter_pairs.append([genuine_idxs[j], imposter_idxs[imposters[1, k]]])

    genuine_pairs = np.array(genuine_pairs)
    imposter_pairs = np.array(imposter_pairs)
    pairs = np.vstack([genuine_pairs, imposter_pairs])
    matches = np.vstack([np.ones((genuine_pairs.shape[0], 1)), np.zeros((imposter_pairs.shape[0], 1))]).squeeze()
    idxs = np.arange(0, matches.shape[0], 1)
    np.random.shuffle(idxs)
    return pairs[idxs], matches[idxs]


def _get_lfw_pairs(filenames, pairs_path, db):
    pairs = []
    matches = []
    prefix = get_prefix(filenames, 2)
    with open(pairs_path) as file:
        lines = file.readlines()
    for line in lines:
        split = line.split('\t')
        name_a, name_b, id_a, id_b = "", "", 0, 0
        genuine = -1
        if len(split) == 3:
            name, id_a, id_b = split
            name_a = name_b = name
            genuine = 1
        elif len(split) == 4:
            name_a, id_a, name_b, id_b = split
            genuine = 0
        filename_a = f"{prefix}/{name_a}/{name_a}_{int(id_a):04d}.jpg"
        filename_b = f"{prefix}/{name_b}/{name_b}_{int(id_b):04d}.jpg"
        try:
            idx_a = np.where(filename_a == filenames)[0][0]
            idx_b = np.where(filename_b == filenames)[0][0]
            pairs.append([idx_a, idx_b])
            matches.append(genuine)
        except:
            pass
    return np.array(pairs), np.array(matches)


def _get_cfp_agedb_pairs(filenames, pairs_path, db):
    pms = np.loadtxt(pairs_path, delimiter=' ')
    pairs = []
    for a, b, m in pms:
        filename_a = f'./{db}/{int(a)}.jpg'
        filename_b = f'./{db}/{int(b)}.jpg'
        idx_a = np.where(filename_a == filenames)[0][0]
        idx_b = np.where(filename_b == filenames)[0][0]
        pairs.append([idx_a, idx_b])
    matches = pms[:, -1]
    return np.array(pairs), matches


def _get_cxlfw_pairs(filenames, pairs_path, db):
    with open(pairs_path) as file:
        lines = file.readlines()
    pairs = []
    matches = []
    for line in lines:
        a, b, m = map(int, line.replace('.jpg', '').split(' '))
        filename_a = f'./{db}/{int(a - 1)}.jpg'
        filename_b = f'./{db}/{int(b - 1)}.jpg'
        idx_a = np.where(filename_a == filenames)[0][0]
        idx_b = np.where(filename_b == filenames)[0][0]
        pairs.append([idx_a, idx_b])
        matches.append(int(m > 0))
    return np.array(pairs), np.array(matches)


def pair(filenames, ids, pairs_path, db):
    pairs = None
    matches = None
    if pairs_path is None:
        pairs, matches = generate_pairs(ids)
    elif db == 'lfw' or db == 'xqlfw':
        pairs, matches = _get_lfw_pairs(filenames, pairs_path, db)
    elif db == 'agedb' or db == 'cfp':
        pairs, matches = _get_cfp_agedb_pairs(filenames, pairs_path, db)
    elif db == 'calfw' or db == 'cplfw':
        pairs, matches = _get_cxlfw_pairs(filenames, pairs_path, db)
    return pairs, matches


class PairDataset:
    def __init__(self, single_dataset: typing.Union[SingleDataset, tuple], pairs_file=None, pairs_function=pair):
        """
        Generates a dataset of genuine and imposter pairs with the underlying single dataset
        :param single_dataset: the dataset to use for generating pairs. Can also be the arguments for constructing a
        SingleDataset.
        :param pairs_file: path to the pairing file which defines all genuine and imposter pairs.
        :param pairs_function: the function which reads the pairs_file and converts it into indices for the single_images.
        """
        if isinstance(single_dataset, tuple):
            single_dataset = SingleDataset(*single_dataset)
        self.embeddings = single_dataset.embeddings
        self.filenames = single_dataset.filenames
        self.db = single_dataset.db
        self.model = single_dataset.model
        self.ids = single_dataset.ids
        self.pairs, self.matches = pairs_function(self.filenames, self.ids, pairs_file, self.db)
