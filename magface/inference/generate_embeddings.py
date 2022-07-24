import os
import cv2
import numpy as np
import torch
import tqdm
from torchvision import transforms
from copy import copy
from torch.utils import data

from magface.inference.network_inf import builder_inf


class ImgInfLoader(data.Dataset):
    def __init__(self, filenames=None, transform=None):
        if type(filenames) == str:
            with open(filenames) as f:
                self.filenames = f.readlines()
        else:
            self.filenames = copy(filenames)
        self.transform = transform

    def __getitem__(self, index):
        ls = self.filenames[index].strip().split()
        # change here
        img_path = ls[0]
        if not os.path.isfile(img_path):
            raise Exception('{} does not exist'.format(img_path))
        img = cv2.imread(img_path)
        if img is None:
            raise Exception('{} is empty'.format(img_path))
        _img = cv2.flip(img, 1)
        return [self.transform(img), self.transform(_img)], img_path

    def __len__(self):
        return len(self.filenames)


def generate_embeddings(filenames, args):
    """
    Generates the single_images for the given
    :param filenames:
    :param args:
    :return:
    """
    filenames = np.array(filenames)
    magface = builder_inf(args)
    magface = torch.nn.DataParallel(magface)
    magface = magface.cuda()
    magface.eval()
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0., 0., 0.],
            std=[1., 1., 1.]),
    ])
    dataset = ImgInfLoader(filenames=filenames, transform=trans)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        shuffle=False)

    embeddings = []
    with torch.no_grad():
        for input_, img_paths in tqdm.tqdm(loader, total=len(loader)):
            embedding = magface(input_[0]).detach().cpu().numpy()
            embeddings.append(embedding)
    return filenames, np.concatenate(embeddings)
