import argparse


def add_magface_args(parser: argparse.ArgumentParser):
    parser.add_argument('-b', '--batch_size', default=16, type=int, metavar='N',
                        help='mini-batch size (default: 16), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--arch', default='iresnet100', type=str, help='backbone architecture')
    parser.add_argument('--cpu_mode', action='store_true', help='Use the CPU.')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to latest checkpoint')
    parser.add_argument('--embedding_size', default=512, type=int, help='The embedding feature size')