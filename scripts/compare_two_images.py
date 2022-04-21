import argparse
from similarity.qmagface import QMagFace
from similarity.cosine import Cosine
from magface.inference.generate_embeddings import generate_embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_a', type=str, help='Path to the first image to compare.')
    parser.add_argument('--file_b', type=str, help='Path to the second image for comparison.')
    parser.add_argument('--resume', type=str, help='Path to the MagFace model checkpoint')
    parser.add_argument('--arch', type=str, default='iresnet100',
                        help='Model architecture. iresnet18, iresnet50 or iresnet100')
    parser.add_argument('--cpu_mode', action='store_true', help='Use the CPU.')
    parser.add_argument('--embedding_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--workers', type=int, default=1)
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
