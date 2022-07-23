import os

import click
import tqdm
import numpy as np
import cv2
from utils.files import list_all_files
from preprocessing.insightface.src.mtcnn_detector import MtcnnDetector
from preprocessing.insightface.src import face_preprocess


def preprocess(det, img):
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


@click.command()
@click.option('--result_dir', '-r', type=click.Path())
@click.option('--source_dir', '-s', type=click.Path())
def main(result_dir, source_dir):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.makedirs(result_dir, exist_ok=True)
    det = MtcnnDetector(
        model_folder=os.path.join(dir_path, '../_models/mtcnn-model/'),
        accurate_landmark=True,
        minsize=50,
        threshold=[0.6, 0.7, 0.8]
    )

    filenames = list_all_files(source_dir)
    filenames_filtered = []
    for filename in tqdm.tqdm(filenames):
        img = cv2.imread(filename)
        p_img = preprocess(det, img)
        if p_img is None:
            continue
        results_filename = filename.replace(source_dir, result_dir)
        os.makedirs(results_filename[:results_filename.rindex('/')], exist_ok=True)
        cv2.imwrite(results_filename, p_img)


if __name__ == '__main__':
    main()
