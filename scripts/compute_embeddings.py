import argparse
from utils.setup import add_magface_args
from utils.files import list_all_files_from_path, list_all_files
from magface.inference.generate_embeddings import generate_embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, help='Path to the preprocessed input images.')
    parser.add_argument('--dst_dir', type=str, help='Path for storing the computed embeddings and filenames.')
    add_magface_args(parser)
    args = parser.parse_args()
    filenames = list_all_files(args.src_dir)
    filenames, embeddings = generate_embeddings(filenames, args)
    print()


if __name__ == '__main__':
    main()
