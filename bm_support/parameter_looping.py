from datahelpers.constants import iden, ye, ai, ps, up, dn, ar, ni, cexp, qcexp, nw, wi, dist, rdist, bdist, pm, \
                                    cpop, cden, ct, affs, aus
from os.path import expanduser, join
import pandas as pd
import numpy as np
from bm_support.add_features import normalize_columns, normalize_columns_with_scaler
from numpy.random import RandomState
from bm_support.sampling import sample_by_length
from datahelpers.dftools import keep_longer_histories
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score,\
    f1_score, classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from bm_support.supervised_aux import simple_stratify
from numpy.random import RandomState
from bm_support.supervised_aux import find_optimal_model, produce_topk_model, produce_topk_model_
from bm_support.supervised_aux import plot_prec_recall, plot_auc, level_corr
from bm_support.supervised_aux import level_corr_one_tail, extract_scalar_metric_from_report
from bm_support.supervised import simple_oversample

from bm_support.beta_est import produce_beta_est, produce_claim_valid
from bm_support.beta_est import produce_beta_interaction_est, produce_interaction_est_weighted
from bm_support.beta_est import produce_interaction_plain


def pop_fold(dfs, k):
    if k < len(dfs):
        df_valid = dfs[k].copy()
        df_train = pd.concat(dfs[:k] + dfs[k+1:])
        return df_train, df_valid
    else:
        return None


def run_experiment(df_dict, available_features, feature_selector, origin='gw', data_mode='t0', len_thr=0,
                   feature_set='claim',
                   model_flag='rf', model_pars={},
                   oversampling_flag=False, n_folds=3, n_trials=10, seed0=13,
                   verbose=False):
    """
    :param df_dict:
    :param origin: gw or lit
    :param data_mode: t0, gt or all
    :param len_thr: 0, 1, 3,
    :param feature_set: claim or batch
    :param model_flag: rf or lr
    :param model_pars:
    :param oversampling_flag: true for oversampling, false otherwise
    :param n_folds: number of folds for validation
    :param n_trials:
    :param seed0: defines sampling for k folds
    :verbose
    :return:
    """

    if data_mode == 'all':
        dfz = pd.concat(df_dict[origin].values())
    else:
        dfz = df_dict[origin][data_mode]
    if len_thr > 0:
        updn_cnt = dfz.groupby([up, dn]).apply(lambda x: x.shape[0])
        good_updns = updn_cnt[updn_cnt > len_thr]
        dfz = dfz.merge(pd.DataFrame(good_updns), left_on=[up, dn], right_index=True)

    nmax = 5000

    rns = RandomState(seed0)
    seeds = rns.randint(nmax, size=n_trials)
    agg = []
    for seed in seeds:
        dfs = sample_by_length(dfz, (up, dn), head=10, seed=seed, frac_test=[1] * n_folds, verbose=verbose)
        if verbose:
            print('seed {0}: sizes {1} from {2}'.format(seed, [x.shape[0] for x in dfs], dfz.shape[0]))
        r = loop_through_kfolds(dfs, available_features, feature_selector, feature_set, model_flag, model_pars,
                              oversampling_flag, seed, verbose)
        agg.append(r)
    return r


def loop_through_kfolds(dfs, available_features, feature_selector,
                        feature_set='claim', model_flag='rf', model_pars={},
                        oversampling_flag=False, seed=11, verbose=False):
    df_reps = []
    dict_reps = []
    case_features = sorted(set(available_features) & set(feature_selector[feature_set]))
    for j in range(len(dfs)):
        df_train, df_test = pop_fold(dfs, j)
        df_rep, dict_rep = model_eval(df_train, df_test, case_features, model_flag,
                                      model_pars, oversampling_flag, seed, verbose)
        df_reps.append(df_rep)
        dict_reps.append(dict_rep)

    # return df_rep, dict_rep
    df_rep_out = merge_dfs('fold', range(len(dfs)), df_reps)
    return df_rep_out, dict_reps


def model_eval(df_train, df_test, case_features, model_flag, model_pars,
               oversampling_flag=False, seed=11, verbose=False):

    scalar_metrics = ['corr', 'auc', 'prec0', 'rec0']
    pos_label = 1

    case_features_red = [c for c in case_features if sum(df_train[c].isnull()) == 0]

    if model_flag != 'rf':
        df_train, scaler = normalize_columns_with_scaler(df_train, case_features_red)
        df_test, scaler = normalize_columns_with_scaler(df_test, case_features_red, scaler)

    if oversampling_flag:
        df_train = simple_oversample(df_train, bdist, seed=seed, ratios=(1, 1))

    X_train, y_train = df_train[case_features_red], df_train[bdist]

    if model_flag == 'rf':
        if not model_pars:
            model_pars = {'min_samples_leaf': 10, 'max_depth': 6, 'random_state': seed}
        clf = RandomForestClassifier(**model_pars)
        clf = clf.fit(X_train, y_train)
    elif model_flag == 'lr':
        if not model_pars:
            model_pars = {'max_features': 8, 'metric_type_foo': accuracy_score}
        clf, c_opt, acc_opt = find_optimal_model(X_train, y_train, **model_pars)
    else:
        return None
    # save claims probs
    # y_probs_train.append(clf.predict_proba(df_train[case_features_red])[:, 1])
    # y_probs_test.append(clf.predict_proba(df_test[case_features_red])[:, 1])

    df_test2 = produce_claim_valid(df_test, case_features_red, clf, pos_label=pos_label)

    test_metrics = produce_interaction_metrics(df_train, df_test2)

    df_train2 = produce_claim_valid(df_train, case_features_red, clf)
    train_metrics = produce_interaction_metrics(df_train, df_train2)

    dfrs = [report_vec2df(irs_, scalar_metrics) for irs_ in [test_metrics, train_metrics]]
    types = ['test', 'train']
    df_report = merge_dfs('dset_type', types, dfrs)
    report_dict = {'test': test_metrics, 'train': train_metrics}

    return df_report, report_dict


def produce_interaction_metrics(df_train, df_test, verbose=False):
    pos_label = 1
    all_cols = []
    ccols = [ps, 'pi_pos']
    iy_probs = produce_beta_interaction_est(df_train, df_test, ccols, verbose=verbose)
    all_cols += ['beta_' + c for c in ccols]

    iy_probs += [produce_interaction_plain(df_test), produce_interaction_plain(df_test, weighted=True)]

    all_cols += ['plain_pi', 'plain_pi_weighted']
    iy_test = df_test[[up, dn, 'bint']].drop_duplicates([up, dn]).set_index([up, dn]).sort_index()['bint']

    irs = [(c, produce_topk_model_(iy_test, iy_prob, pos_label=pos_label)) for c, iy_prob in zip(all_cols, iy_probs)]
    return irs


def mdict2df(mdict, scalar_metrics):
    return pd.DataFrame([mdict[c] for c in scalar_metrics], index=scalar_metrics).T


def merge_dfs(column, values, dfs):
    for df, val in zip(dfs, values):
        df[column] = val
    df = pd.concat(dfs)
    return df


def report_vec2df(irs, scalar_metrics):
    dfs = [mdict2df(dd, scalar_metrics) for _, dd in irs]
    vals = [v for v, _ in irs]
    df = merge_dfs('estimator', vals, dfs)
    return df

