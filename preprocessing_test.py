import os
import time
from contextlib import contextmanager
import tqdm
import sys
import numpy as np
import cv2
import torch
import torchvision.transforms
from collections import namedtuple
from datasets.single import SingleDataset
from utils.files import list_all_files
from similarity.cosine import Cosine
from preprocessing.magface.network_inf import builder_inf
from preprocessing.insightface.src.mtcnn_detector import MtcnnDetector
from preprocessing.insightface.src import face_preprocess


def preprocess(det, img):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detected = det.detect_face(img, det_type=0)
    if detected is None:
        return None

    bbox, points = detected
    if bbox.shape[0] == 0:
        return None

    points = points[0, :].reshape((2, 5)).T
    image = face_preprocess.preprocess(img, bbox, points, image_size="112,112")
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def main():
    result_dir = '_data/single_images/magface100_recomp_3/'
    os.makedirs(result_dir, exist_ok=True)
    det = MtcnnDetector(
        model_folder='_models/mtcnn-model/',
        accurate_landmark=True,
        minsize=50,
        threshold=[0.6, 0.7, 0.8]
    )
    Args = namedtuple('Args', ['arch', 'resume', 'embedding_size', 'cpu_mode'])
    args = Args('iresnet100', '_models/magface_models/magface_epoch_00025.pth', 512, True)
    model = builder_inf(args)
    model = torch.nn.DataParallel(model)
    model.eval()

    trans = torchvision.transforms.ToTensor()

    filenames = list_all_files('_data/images/lfw/')
    filenames_filtered = []
    embeddings = []
    for filename in tqdm.tqdm(filenames):
        img = cv2.imread(filename)
        p_img = preprocess(det, img)
        if p_img is None:
            continue

        input_ = torch.unsqueeze(trans(p_img), 0).to('cuda')
        embeddings.append(model(input_).detach().cpu().numpy().squeeze())
        filenames_filtered.append(filename)

    embeddings = np.stack(embeddings)
    filenames = np.array(filenames_filtered)
    np.save(result_dir + 'embeddings_lfw.npy', embeddings)
    np.save(result_dir + 'filenames_lfw.npy', filenames)
    computed = SingleDataset(result_dir, 'lfw')
    test = SingleDataset('_data/single_images/magface100', 'lfw')

    print()


if __name__ == '__main__':
    main()
