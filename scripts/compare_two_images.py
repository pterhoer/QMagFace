import argparse
from similarity.qmagface import QMagFace
from similarity.cosine import Cosine
from magface.inference.generate_embeddings import generate_embeddings
from utils.setup import add_magface_args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_a', type=str, help='Path to the first image to compare.')
    parser.add_argument('--file_b', type=str, help='Path to the second image for comparison.')
    add_magface_args(parser)
    args = parser.parse_args()
    filenames = [args.file_a, args.file_b]
    _, embs = generate_embeddings(filenames, args)

    simi = QMagFace(0.077428, 0.125926)
    simi_ = Cosine()
    emb_a = embs[0].reshape(1, -1)
    emb_b = embs[1].reshape(1, -1)
    print(simi.similarity(emb_a, emb_b))
    print(simi_.similarity(emb_a, emb_b))


if __name__ == '__main__':
    main()
