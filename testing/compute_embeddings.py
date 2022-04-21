import argparse
import numpy as np
from utils.setup import add_magface_args
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
    parser.add_argument('--src_dir', type=str, metavar='PATH', help='path to the source image directory')
    parser.add_argument('--int_dir', type=str, metavar='PATH', help='path to save preprocessed numpy images')
    parser.add_argument('--emb_file', type=str, metavar='PATH', help='file to save computed embeddings')
    add_magface_args(parser)
    args_ = parser.parse_args()
    main(args_)
