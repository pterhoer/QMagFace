import argparse
from preprocessing.preprocessing import align_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, help='Source directory for preprocessing.')
    parser.add_argument('--dst_dir', type=str, help='Destination directory for preprocessing.')
    args_ = parser.parse_args()
    dst_filenames = align_images(args_.src_dir, args_.dst_dir)


if __name__ == '__main__':
    main()
