import argparse

from bm_support.supervised import logreg_driver


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # P(C| E) with time features, density and cur den
    #     origin = 'gw'
    #     batchsize = 1
    #     a = 0.0
    #     b = 1.0
    #     version = 11
    #     cutoff_len = 1
    # an_version = 4
    # hash_int = 775498

    parser.add_argument('-o', '--origin',
                        default='gw',
                        help='type of data to work with [gw, lit]')

    parser.add_argument('-v', '--version',
                        default=8, type=int,
                        help='version of data source')

    parser.add_argument('-p', '--partition-sequence',
                        nargs='+', default=[0.0, 1.0], type=float,
                        help='define interval of observed freqs for sequence consideration')

    parser.add_argument('-c', '--cutoff-len',
                        default=1, type=int,
                        help='version of data source')

    parser.add_argument('-b', '--nbatches',
                        default=1, type=int,
                        help='size of data batches')

    parser.add_argument('--hash',
                        default=0, type=int,
                        help='required: hash of dataset file')

    parser.add_argument('-d', '--depth', type=int,
                        default=3,
                        help='test on the head of the dataset')

    parser.add_argument('--test', type=int,
                        default=-1,
                        help='test on the head of the dataset')

    args = parser.parse_args()
    print(args)

    low_f, hi_f = args.partition_sequence

    logreg_driver(args.origin, args.version, args.nbatches, args.cutoff_len,
                  low_f, hi_f, args.hash, args.depth)
