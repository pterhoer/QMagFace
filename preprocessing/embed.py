import os

import click
import tqdm
import numpy as np
import cv2
import torch
import torchvision.transforms
from torch.utils import data
from collections import namedtuple
from utils.files import list_all_files
from preprocessing.magface.network_inf import builder_inf


class ImgDataset(data.Dataset):
    def __init__(self, filenames, transform):
        super(ImgDataset, self).__init__()
        self.filenames = filenames
        self.transform = transform

    def __getitem__(self, item):
        path = self.filenames[item]
        img = cv2.imread(path)
        return self.transform(img), path

    def __len__(self):
        return len(self.filenames)


@click.command()
@click.option('--result_dir', '-r', type=click.Path())
@click.option('--source_dir', '-s', type=click.Path())
@click.option('--model_path', '-m', type=click.Path(), default='_models/magface_models/magface_epoch_00025.pth')
@click.option('--db', '-d', type=str)
def main(result_dir, source_dir, model_path, db):
    os.makedirs(result_dir, exist_ok=True)
    Args = namedtuple('Args', ['arch', 'resume', 'embedding_size', 'cpu_mode'])
    args = Args('iresnet100', model_path, 512, True)
    model = builder_inf(args)
    model = torch.nn.DataParallel(model)
    model.eval()

    trans = torchvision.transforms.ToTensor()
    filenames = list_all_files(source_dir)
    dataset = ImgDataset(filenames, trans)
    loader = data.DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True, shuffle=False)
    embeddings = torch.tensor([], dtype=torch.float32)
    with torch.no_grad():
        for input_, label in tqdm.tqdm(loader):
            input_ = input_.to('cuda')
            embedding = model(input_).to('cpu')
            embeddings = torch.cat([embeddings, embedding])

    embeddings = embeddings.numpy()
    filenames = np.array(filenames)
    np.save(os.path.join(result_dir, f'embeddings_{db}.npy'), embeddings)
    np.save(os.path.join(result_dir, f'filenames_{db}.npy'), filenames)


if __name__ == '__main__':
    main()
