import os.path
import re

import numpy as np

import utils.files


def _get_generic_labels(filenames, split, split_idx):
    labels = np.array([int(re.split(r'[/\\]', path)[-1].split(split)[split_idx]) for path in filenames])
    return labels


def _get_lfw_labels(filenames):
    previous_names = set()
    label = 0
    labels = []
    for path in filenames:
        name = path[:-9]
        if name not in previous_names:
            label += 1
            previous_names.add(name)
        labels.append(label)
    return np.array(labels)


def filenames_to_ids(filenames, db):
    ids = None
    if db == 'lfw':
        ids = _get_lfw_labels(filenames)
    elif db == 'morph':
        ids = _get_generic_labels(filenames, '_', 0)
    elif db == 'colorferet':
        ids = _get_generic_labels(filenames, '_', 0)
    elif db == 'adience':
        ids = _get_generic_labels(filenames, '.', 1)
    return ids


class SingleDataset:
    def __init__(self, root, db, filename_to_id_func=filenames_to_ids, model=''):
        """
        Loads the necessary files to create a dataset of single_images and person ids.
        :param root: root to the location where the numpy files are saved. Each db consists of 2 files, the filenames
        file and the single_images file, the names should then be filenames_[db].npy and embeddings_[db].npy.
        :param db: The name of the image database.
        :param filename_to_id_func: if you have a different filename structure than the one that is assumed by the
        implemented function, you can code your own function which converts filenames into ids for each person.
        :param model: Optional parameter to save the name of the model which computed the single_images.
        """
        self.embeddings = np.load(os.path.join(root, f'embeddings_{db}.npy'))
        self.filenames = np.load(os.path.join(root, f'filenames_{db}.npy'))
        self.filenames = np.array(list(map(utils.files.convert_backslashes, self.filenames)))
        self.db = db
        self.model = model
        self.ids = filename_to_id_func(self.filenames, db)


def test():
    ds = SingleDataset('../_data/magface100', 'lfw')
    print()


if __name__ == '__main__':
    test()
