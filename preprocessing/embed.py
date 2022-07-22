import os

import click
import tqdm
import numpy as np
import cv2
import torch
import torchvision.transforms
from collections import namedtuple
from datasets.single import SingleDataset
from utils.files import list_all_files
from preprocessing.magface.network_inf import builder_inf


@click.command()
@click.option('--result_dir', '-r', type=click.Path())
@click.option('--source_dir', '-s', type=click.Path())
def main(result_dir, source_dir):
    os.makedirs(result_dir, exist_ok=True)
    Args = namedtuple('Args', ['arch', 'resume', 'embedding_size', 'cpu_mode'])
    args = Args('iresnet100', '../_models/magface_models/magface_epoch_00025.pth', 512, True)
    model = builder_inf(args)
    model = torch.nn.DataParallel(model)
    model.eval()

    trans = torchvision.transforms.ToTensor()

    filenames = list_all_files(source_dir)
    filenames_filtered = []
    embeddings = []
    for filename in tqdm.tqdm(filenames):
        p_img = cv2.imread(filename)

        input_ = torch.unsqueeze(trans(p_img), 0).to('cuda')
        embeddings.append(model(input_).detach().cpu().numpy().squeeze())
        filenames_filtered.append(filename)

    embeddings = np.stack(embeddings)
    filenames = np.array(filenames_filtered)
    np.save(result_dir + 'embeddings_lfw.npy', embeddings)
    np.save(result_dir + 'filenames_lfw.npy', filenames)


if __name__ == '__main__':
    main()
