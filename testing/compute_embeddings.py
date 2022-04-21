import argparse
import numpy as np
from utils.files import list_all_files_from_path
from preprocessing.preprocessing import align_images
from magface.inference.generate_embeddings import generate_embeddings

from similarity.cosine import Cosine


def main(args):
    sim = Cosine()
    embs = np.load('_data/datasets/lfw_emb.npy')
    fns = np.load('_data/datasets/lfw_fns.npy')
    oembs = np.load('_data/magface100/embeddings_lfw.npy')
    ofns = np.load('_data/magface100/filenames_lfw.npy')
    sims = sim.similarity(embs, oembs)
    print('Aligning Images')
    aligned_filenames = align_images(args.src_dir, args.int_dir)
    filenames, embeddings = generate_embeddings(aligned_filenames, args)
    np.save(args.emb_file, embeddings)
    np.save(args.emb_file.replace('emb', 'fns'), filenames)
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', default='iresnet100', type=str, help='backbone architecture')
    parser.add_argument('--cpu_mode', action='store_true', help='Use the CPU.')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to latest checkpoint')
    parser.add_argument('--embedding_size', default=512, type=int, help='The embedding feature size')
    parser.add_argument('--src_dir', type=str, metavar='PATH', help='path to the source image directory')
    parser.add_argument('--int_dir', type=str, metavar='PATH', help='path to save preprocessed numpy images')
    parser.add_argument('--emb_file', type=str, metavar='PATH', help='file to save computed embeddings')
    parser.add_argument('-b', '--batch_size', default=16, type=int, metavar='N',
                        help='mini-batch size (default: 16), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    args_ = parser.parse_args()
    main(args_)
