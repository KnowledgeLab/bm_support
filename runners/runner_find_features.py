from datahelpers.constants import iden, ye, ai, ps, up, dn, ar, ni, cexp, qcexp, nw, wi, dist, rdist, pm, \
                                    cpop, cden, ct, affs, aus
from os.path import expanduser, join
import pandas as pd
from bm_support.add_features import generate_feature_groups
from bm_support.add_features import normalize_columns
from bm_support.supervised_aux import study_sample, metric_selector
from functools import partial
from multiprocessing import Pool
from copy import deepcopy
import warnings
from numpy.random import RandomState
import gzip
import pickle
import argparse

warnings.filterwarnings('ignore')


def run(origin, version, an_version, model_type, n_trials, n_subtrials, n_estimators, datapath=None,
        seed0=13, n_jobs=1, verbose=False):

    # an_version = 13
    # origin = 'lit'
    # version = 8

    # origin = 'gw'
    # version = 11

    # model_type = 'rf'
    # seed0 = 17
    # n_trials = 50
    # n_subtrials = 10
    # n_estimators = 17
    # n_estimators = 55

    min_log_alpha = -2
    max_log_alpha = 2
    log_reg_dict = {'min_log_alpha': -2., 'max_log_alpha': 2.}

    eps = 0.2
    upper_exp, lower_exp = 1 - eps, eps
    # thrs = [-1e-8, lower_exp, upper_exp, 1.0001e0]
    if datapath:
        col_families = generate_feature_groups(expanduser(join(datapath, 'v12_columns.txt')))
    else:
        col_families = generate_feature_groups(expanduser('~/data/kl/columns/v12_columns.txt'))

    if verbose:
        print('Number of col families: {0}. Keys: {1}'.format(len(col_families), sorted(col_families.keys())))

    col_families = {k: v for k, v in col_families.items() if 'future' not in k}
    if verbose:
        print('Number of col families (excl. future): {0}. Keys: {1}'.format(len(col_families),
                                                                             sorted(col_families.keys())))

    columns_interest = [x for sublist in col_families.values() for x in sublist]
    if datapath:
        df_path = expanduser(join(datapath, '{0}_{1}_{2}.h5'.format(origin, version, an_version)))
    else:
        df_path = expanduser('~/data/kl/final/{0}_{1}_{2}.h5'.format(origin, version, an_version))
    df = pd.read_hdf(df_path, key='df')

    # mask: literome - mask out a specific interaction
    mask_lit = (df[up] == 7157) & (df[dn] == 1026)

    # mask:  interaction with more than 3 claims
    thr = 3
    mask_len_ = (df.groupby(ni).apply(lambda x: x.shape[0]) > thr)
    mask_max_len = df[ni].isin(mask_len_[mask_len_].index)
    print(mask_max_len.shape[0], sum(mask_max_len))

    # mask : interactions which are between
    eps_window_mean = 0.1
    mean_col = 0.5
    mask_exp = ((df[cexp] <= lower_exp - eps_window_mean) | (df[cexp] >= upper_exp + eps_window_mean)
                # | ((df2[cexp] <= (mean_col + eps_window_mean)) & (df2[cexp] >= (mean_col - eps_window_mean)))
                )

    feature_dict = deepcopy(col_families)
    # families = ['affiliations_affind', 'affiliations_comm_size', 'affiliations_ncomms',
    #             'affiliations_ncomponents', 'affiliations_size_ulist', 'affiliations_suppind',
    #             'ai', 'ar', 'authors_affind', 'authors_comm_size', 'authors_ncomms', 'authors_ncomponents',
    #             'authors_size_ulist', 'authors_suppind', 'cden', 'citations',
    #             'cite_count', 'cpop', 'delta_year', 'ksst', 'lincscomm_size', 'lincssame_comm',
    #             'litgwcsize_dn', 'litgwcsize_up',
    #             'litgweff_comm_size', 'litgwsame_comm', 'nhi', 'past_affind',
    #             'past_comm_size', 'past_ncomms', 'past_ncomponents',
    #             'past_size_ulist', 'past_suppind', 'pre_affs', 'pre_authors', 'time']
    families = ['affiliations_comm_size',
                'ai', 'ar', 'cden', 'citations',
                'cite_count', 'cpop', 'delta_year', 'ksst', 'lincscomm_size', 'lincssame_comm',
                'litgweff_comm_size', 'litgwsame_comm', 'nhi', 'past_affind',
                'past_comm_size', 'time']

    feature_dict = {k: v for k, v in feature_dict.items() if k in families}

    trial_features = [x for sublist in feature_dict.values() for x in sublist]

    feature_dict_inv = {}
    for k, v in feature_dict.items():
        feature_dict_inv.update({x: k for x in v})

    # mask: not nulls in trial features
    masks = []
    for c in trial_features:
        masks.append(df[c].notnull())

    mask_notnull = masks[0]
    for m in masks[1:]:
        mask_notnull &= m

    print('Experimental mask len {0}'.format(sum(mask_exp)))
    print('Number of trial features: {0}'.format(len(trial_features)))
    print('Number of notnull entries (over all features): {0} from {1}'.format(sum(mask_notnull), mask_notnull.shape))

    if origin == 'lit':
        mask_agg = mask_notnull & ~mask_lit
    else:
        mask_agg = mask_notnull

    dfw = df.loc[mask_agg].copy()

    #metric to optimize for
    mm = 'precision'

    nmax = 5000

    meta_report = []

    rns = RandomState(seed0)
    seeds = rns.randint(nmax, size=n_trials)
    cnt = 0

    if model_type == 'lr':
        dfw = normalize_columns(dfw, trial_features)

    func = partial(study_sample, dfw=dfw, dist=dist, feature_dict=feature_dict, metric_mode=mm,
                   model_type=model_type,
                   n_subtrials=n_subtrials, n_estimators=n_estimators,
                   log_reg_dict=log_reg_dict, verbose=verbose)

    if n_jobs > 1:
        with Pool(n_jobs) as p:
            meta_report = p.map(func, seeds)
    else:
        meta_report = list(map(func, seeds))

    if datapath:
        fout = expanduser(join(datapath, '{0}_{1}_{2}_{3}_seed0_{4}.report.pgz'.format(origin, version,
                                                                                       an_version, model_type, seed0)))
    else:
        fout = expanduser('~/data/kl/reports/{0}_{1}_{2}_{3}_seed0_{4}.report.pgz'.format(origin, version,
                                                                                          an_version,
                                                                                          model_type, seed0))

    with gzip.open(fout, 'wb') as fp:
        pickle.dump(meta_report, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--origin',
                        default='lit',
                        help='type of data to work with [gw, lit]')

    parser.add_argument('-v', '--version',
                        default=8, type=int,
                        help='version of data source')

    parser.add_argument('-a', '--anversion',
                        default=12, type=int,
                        help='version of data source')

    parser.add_argument('-m', '--model_type',
                        default='rf',
                        help='type of model to study')

    parser.add_argument('-t', '--ntrials',
                        default=3, type=int,
                        help='size of data batches')

    parser.add_argument('-st', '--subtrials',
                        default=2, type=int,
                        help='version of data source')

    parser.add_argument('-e', '--estimators',
                        default=11, type=int,
                        help='number of trees in a forest')

    parser.add_argument('-p', '--parallel',
                        default=1, type=int,
                        help='number of cores to be used')

    parser.add_argument('-s', '--seed0',
                        default=17, type=int,
                        help='number of cores to be used')

    parser.add_argument('--verbosity',
                        default=True, type=bool,
                        help='True for verbose output ')

    parser.add_argument('--datapath',
                        default=None, type=str,
                        help='True for verbose output ')

    args = parser.parse_args()
    run(args.origin, args.version, args.anversion, args.model_type,
        args.ntrials, args.subtrials, args.estimators, args.datapath, args.seed0,
        args.parallel, args.verbosity)
