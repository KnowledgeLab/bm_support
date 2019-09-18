from datahelpers.constants import up, dn
import argparse
from bm_support.add_features import define_laststage_metrics, prepare_datasets
from os.path import expanduser, join
from numpy.random import RandomState
from bm_support.supervised_aux import run_model_iterate_over_datasets
import pickle
import gzip


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--mode',
                        default='posneg',
                        help='type of data to work with [gw, lit]')

    parser.add_argument('-n', '--niter',
                        type=int,
                        default=1,
                        help='number of iterations')

    parser.add_argument('-s', '--seed',
                        type=int,
                        default=13,
                        help='seed, used to control random functions')

    parser.add_argument('-o', '--oversample',
                        type=bool,
                        default=False,
                        help='use if your dataset is unbalanced')

    parser.add_argument('-t', '--thr',
                        type=int,
                        default=0,
                        help='threshold in length')

    min_leaf_frac_baseline = 0.005

    args = parser.parse_args()
    predict_mode = args.mode
    seed = args.seed
    len_thr = args.thr
    n_iter = args.niter
    version = n_iter

    oversample = args.oversample

    print(f'mode: {predict_mode}')
    mode = 'rf'
    rns = RandomState(seed)

    df_dict, cfeatures, target = prepare_datasets(predict_mode, len_thr)

    verbose = False
    if predict_mode == 'neutral':
        oversample = True
    else:
        oversample = False

    report = []
    clf_parameters = {'max_depth': 2, 'n_estimators': 100}
    extra_parameters = {'min_samples_leaf_frac': min_leaf_frac_baseline}

    container = run_model_iterate_over_datasets(df_dict, cfeatures, rns,
                                                target=target, mode=mode, n_splits=3,
                                                clf_parameters=clf_parameters,
                                                extra_parameters=extra_parameters,
                                                n_iterations=n_iter,
                                                oversample=oversample)

    fpath = expanduser('~/data/kl/reports/')

    with gzip.open(fpath + f'models_{predict_mode}_thr_{len_thr}_v{version}.pkl.gz', 'wb') as f:
        pickle.dump(container, f)
