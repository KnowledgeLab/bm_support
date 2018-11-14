from datahelpers.constants import iden, ye, ai, ps, up, dn, ar, ni, cexp, qcexp, nw, wi, dist, rdist, pm, \
                                    cpop, cden, ct, affs, aus
from os.path import expanduser
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


def run(origin, version, an_version, model_type, n_trials, n_subtrials, n_estimators, seed0, n_jobs, verbose):

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

    col_families = generate_feature_groups(expanduser('~/data/kl/columns/v12_columns.txt'))
    if verbose:
        print('Number of col families: {0}. Keys: {1}'.format(len(col_families), sorted(col_families.keys())))

    col_families = {k: v for k, v in col_families.items() if 'future' not in k}
    if verbose:
        print('Number of col families (excl. future): {0}. Keys: {1}'.format(len(col_families),
                                                                             sorted(col_families.keys())))

    columns_interest = [x for sublist in col_families.values() for x in sublist]

    df = pd.read_hdf(expanduser('~/data/kl/final/{0}_{1}_{2}.h5'.format(origin, version, an_version)), key='df')

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

    # for seed in seeds:
    #     report_dict = study_sample(seed, dfw, dist, feature_dict,
    #              mm, model_type, n_subtrials, n_estimators,
    #              log_reg_dict, verbose)

        # rns = RandomState(seed)
        # df_train, df_testgen = train_test_split(dfw, test_size=0.4,
        #                                         random_state=rns, stratify=dfw[dist])
        #
        # df_valid, df_test = train_test_split(df_testgen, test_size=0.5,
        #                                      random_state=rns)
        #
        # vc = df_train[dist].value_counts()
        # if verbose and vc.shape[0] < 5:
        #     print('*** df_train dist vc')
        #     print(vc)
        # # training on the normalized frequencies
        # df_train2 = simple_stratify(df_train, dist, seed, ratios=(2, 1, 1))
        #
        # if model_type == 'rf':
        #     param_dict = {'n_estimators': n_estimators, 'max_features': None, 'n_jobs': n_jobs}
        # else:
        #     param_dict = {}
        #
        # meta_agg = []
        # if model_type == 'rf':
        #     enums = rns.randint(nmax, size=n_subtrials)
        # elif model_type == 'lr':
        #     delta = (max_log_alpha - min_log_alpha) / n_subtrials
        #     enums = 1e1 ** np.arange(min_log_alpha, max_log_alpha, delta)
        # else:
        #     enums = []
        #
        # for ii in enums:
        #     if model_type == 'rf':
        #         param_dict['random_state'] = ii
        #     elif model_type == 'lr':
        #         param_dict['C'] = ii
        #
        #     # for random forest different seed yield different models, for logreg models are penalty-dependent
        #     cfeatures, cmetrics, cvector_metrics, model_ = select_features_dict(df_train2, df_test, dist,
        #                                                                         feature_dict,
        #                                                                         model_type=model_type,
        #                                                                         max_features_consider=8,
        #                                                                         metric_mode=mm,
        #                                                                         model_dict=param_dict,
        #                                                                         verbose=verbose)
        #     vscalar, vvector = report_metrics(model_, df_valid[cfeatures], df_valid[dist])
        #     ii_dict = {}
        #     ii_dict['run_par'] = seed
        #     ii_dict['current_features'] = cfeatures
        #     ii_dict['current_metrics'] = cmetrics
        #     ii_dict['current_vector_metrics'] = cvector_metrics
        #     ii_dict['validation_scalar_metrics'] = vscalar
        #     ii_dict['validation_vector_metrics'] = vvector
        #     ii_dict['model'] = model_
        #     if model_type == 'lr':
        #         ii_dict['pval_errors'] = logit_pvalue(model_, df_train2[cfeatures])
        #
        #     meta_agg.append(ii_dict)
        #
        # vscalar_mm = [x['validation_scalar_metrics'][metric_selector[mm]] for x in meta_agg]
        # vscalar = [x['validation_scalar_metrics'] for x in meta_agg]
        # vvector = [x['validation_vector_metrics'] for x in meta_agg]
        # index_best_run = np.argmax(vscalar_mm)
        # best_features = meta_agg[index_best_run]['current_features']
        # best_feature_groups = [feature_dict_inv[f] for f in best_features]
        #
        # report_dict = {}
        # report_dict['best_features'] = best_features
        # report_dict['best_feature_groups'] = best_feature_groups
        # report_dict['max_scalar_mm'] = np.max(vscalar_mm)
        # report_dict['vscalar'] = vscalar
        # report_dict['vvector'] = vvector
        # report_dict['corr_all'] = dfw[best_features + [dist]].corr()[dist]
        # report_dict['corr_all_test'] = df_test[best_features + [dist]].corr()[dist]
        # report_dict['corr_all_valid'] = df_valid[best_features + [dist]].corr()[dist]
        # if model_type == 'lr':
        #     report_dict['pval_errors'] = meta_agg[index_best_run]['pval_errors']
        #
        # meta_report.append(report_dict)
        # print('{0:.2f} % done.'.format(100*(cnt+1)/len(seeds)))
        # cnt += 1
        # fig, ax = plt.subplots(figsize=(5, 5))
        # ax.set_ylabel(mm)
        # for item in meta_agg:
        #     seed, cfeatures, cmetrics, cvector_metrics, vscalar, vvector, model_rf = item
        #     xcoords = list(range(len(cmetrics)))
        #     ax.plot(xcoords, np.array(cmetrics)[:, metric_selector[mm]])

    fout = expanduser('~/data/kl/reports/{0}_{1}_{2}_{3}_seed0_{4}.report.pgz'.format(origin, version,
                                                                                      an_version, model_type, seed0))
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

    args = parser.parse_args()
    run(args.origin, args.version, args.anversion, args.model_type,
        args.ntrials, args.subtrials, args.estimators, args.seed0, args.parallel, args.verbosity)
