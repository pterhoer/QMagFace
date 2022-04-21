# Preprocessing of Images

# This script provides image the preprocessing function using MTCNN 
# Parts of this file are strongly adapted from:
# https://github.com/deepinsight/insightface/blob/master/src/common/face_preprocess.py   

# Author: Marco Huber, 2020
# Fraunhofer IGD
# marco.huber[at]igd.fraunhofer.de
import multiprocessing
import os

import cv2
import numpy as np
from utils.files import path_join, convert_backslashes, list_all_files_from_path
from mtcnn import MTCNN
from skimage import transform
from tqdm import tqdm


def setup_img(img):
    """
    Prepares the input image

    Parameters
    ----------
    img : img array
        The preprocessed and aligned keras model.

    Returns
    -------
    in_img : TYPE
        Prepared image ready to be fed into the network.

    """

    # preprare image
    in_img = preprocess_img(img)

    if in_img is None:
        return None

    in_img = np.expand_dims(in_img, axis=0)
    in_img = np.moveaxis(in_img, 1, 3)

    return in_img


def preprocess_img(img, detector=None):
    """
    Aligns and preprocess the provided image

    Parameters
    ----------
    img : array of the images
        The image to be aligned and preprocessed.

    Returns
    -------
    nimg : numpy ndarray
        Aligned and processed image.

    """
    # define thresholds
    thrs = [0.6, 0.7, 0.8]

    # get detector
    if detector is None:
        detector = MTCNN(steps_threshold=thrs)

    # detect face
    detected = detector.detect_faces(img)

    if detected is None or detected == []:
        print("MTCNN could not detected a face.")
        return None

    # get box and points
    bbox, points = detected[0]['box'], detected[0]['keypoints']

    # rearrange points
    p_points = []
    for v in points.values():
        p_points.append(v)

    p_points = np.asarray(p_points)

    # preprocess
    nimg = preprocess(img, bbox, p_points, image_size="112,112")
    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)

    # return np.transpose(nimg, (2,0,1))
    return nimg


def read_image(img_path, **kwargs):
    mode = kwargs.get('mode', 'rgb')
    layout = kwargs.get('layout', 'HWC')

    if mode == 'gray':
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if mode == 'rgb':
            img = img[..., ::-1]
        if layout == 'CHW':
            img = np.transpose(img, (2, 0, 1))
    return img


def preprocess(img, bbox=None, landmark=None, **kwargs):
    if isinstance(img, str):
        img = read_image(img, **kwargs)

    M = None
    image_size = []
    str_image_size = kwargs.get('image_size', '')

    if len(str_image_size) > 0:
        image_size = [int(x) for x in str_image_size.split(',')]
        if len(image_size) == 1:
            image_size = [image_size[0], image_size[0]]
        assert len(image_size) == 2
        assert image_size[0] == 112 or image_size[0] == 160
        assert image_size[0] == 112 or image_size[1] == 96 or image_size[0] == 160

    if landmark is not None:
        assert len(image_size) == 2
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        if image_size[1] == 112 or image_size[1] == 160:
            src[:, 0] += 8.0
        dst = landmark.astype(np.float32)

        tform = transform.SimilarityTransform()
        tform.estimate(dst, src)
        M = tform.params[0:2, :]

    if M is None:
        if bbox is None:
            det = np.zeros(4, dtype=np.int32)
            det[0] = int(img.shape[1] * 0.0625)
            det[1] = int(img.shape[0] * 0.0625)
            det[2] = img.shape[1] - det[0]
            det[3] = img.shape[0] - det[1]
        else:
            det = bbox
        margin = kwargs.get('margin', 44)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img.shape[1])
        bb[3] = np.minimum(det[3] + margin / 2, img.shape[0])
        ret = img[bb[1]:bb[3], bb[0]:bb[2], :]
        if len(image_size) > 0:
            ret = cv2.resize(ret, (image_size[1], image_size[0]))
        return ret
    else:
        assert len(image_size) == 2
        warped = cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue=0.0)
        return warped


def _alignment_process(*args):
    src_filenames, dst_filenames = args
    detector = MTCNN(steps_threshold=[0.6, 0.7, 0.8])
    for image_src_path, image_dst_path in tqdm(zip(src_filenames, dst_filenames), total=len(src_filenames)):
        image = read_image(image_src_path)
        network_input = preprocess_img(image, detector)
        os.makedirs(os.path.dirname(image_dst_path), exist_ok=True)
        cv2.imwrite(image_dst_path, network_input)
    return


def align_images(src_path_prefix, dst_path_prefix, filename_conversion_func=str):
    """
    Aligns all images in list_of_filenames and saves them to dst_path_prefix.
    Note: this method creates a subprocess to isolate the tensorflow instance.
    :param src_path_prefix: the directory which holds all images.
    :param dst_path_prefix: the directory to which the aligned images should be saved.
    :param filename_conversion_func: a function which takes in the source filename and converts it to a different
    filename, for example to remove redundant directories. For simplicity, any backslashes in filenames get converted to
    forward slashes, before going through filename_conversion_func.
    :return: a list of filenames of the aligned images.
    """
    list_of_filenames = list_all_files_from_path(src_path_prefix)
    src_filenames = []
    dst_filenames = []
    dst_filenames_no_prefix = []
    for filename in list_of_filenames:
        filename = convert_backslashes(filename)
        dst_filename = filename_conversion_func(filename)
        src_filenames.append(path_join(src_path_prefix, filename))
        dst_filenames.append(path_join(dst_path_prefix, dst_filename))
        dst_filenames_no_prefix.append(dst_filename)

    process = multiprocessing.Process(target=_alignment_process, args=(src_filenames, dst_filenames))
    process.start()
    process.join()
    return dst_filenames
