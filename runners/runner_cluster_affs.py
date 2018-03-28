import argparse
from os.path import expanduser, join
import gzip
import pickle
from bm_support.add_features import retrieve_wos_aff_au_df, process_affs
from bm_support.add_features import cluster_affiliations
import pandas as pd
from datahelpers.constants import pm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--sourcepath',
                        default='~/data/wos/',
                        help='path to input data')

    parser.add_argument('-d', '--destpath', default='.',
                        help='path to output data')

    parser.add_argument('-n', '--nproc',
                        default=1, type=int,
                        help='number of threads')

    parser.add_argument('--pm-fname', default='pmids.csv.gz',
                        help='filename of the input gzip compressed csv with pmids')

    parser.add_argument('--test',
                        type=int, default=-1,
                        help='test on the head of the dataset')

    args = parser.parse_args()
    print(args)

    spath = expanduser(args.sourcepath)
    dpath = expanduser(args.destpath)
    pm_fname = args.pm_fname

    # has format (index, pm) with top row being ",pmid"
    dfp = pd.read_csv(join(spath, pm_fname), compression='gzip', index_col=0)
    pmids = list(set(dfp[pm].unique()))

    df_wos = retrieve_wos_aff_au_df(spath)
    index_affs, pm_aff_phrases, a2i = process_affs(df_wos, pmids)

    if args.test > 0:
        pmids = pmids[:args.test]

    pm2c, aff_dict = cluster_affiliations(df_wos, pmids, n_processes=args.nproc, debug=True)

    pm2c_fname = join(dpath, 'pm2id_dict.pgz')

    with gzip.open(pm2c_fname, 'wb') as fp:
        pickle.dump(pm2c, fp)

    aff_dict_fname = join(dpath, 'id2affs_dict.pgz')

    with gzip.open(aff_dict_fname, 'wb') as fp:
        pickle.dump(aff_dict, fp)
