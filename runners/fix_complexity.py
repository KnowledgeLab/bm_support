import argparse
from bm_support.add_features import generate_feature_groups, define_laststage_metrics
from bm_support.add_features import define_laststage_metrics
from copy import deepcopy
from os.path import expanduser, join
from numpy.random import RandomState
import numpy as np
import json
from bm_support.supervised_aux import run_neut_models
from bm_support.math import interpolate_nonuniform_linear, integral_linear, get_function_values, find_bbs
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def prepare_datasets(predict_mode_='posneg'):
    fprefix = f'predict_{predict_mode_}'

    # model_type = 'lr'
    model_type = 'rf'

    if model_type == 'rf':
        forest_flag = True
    else:
        forest_flag = False

    fname = expanduser('~/data/kl/columns/feature_groups_v3.txt')
    with open(fname, 'r') as f:
        feat_selector = json.load(f)

    df_dict = {}

    for origin in ['gw', 'lit']:
        df_dict[origin] = define_laststage_metrics(origin, predict_mode=predict_mode_, verbose=True)
        print(f'>>> {origin} {predict_mode_} {df_dict[origin].shape[0]}')

    if predict_mode_ == 'neutral' or predict_mode_ == 'posneg':
        selectors = ['interaction']
        target_ = 'bint'

        cfeatures_ = ['mu*', 'mu*_pct', 'mu*_absmed', 'mu*_absmed_pct',
                      'degree_source', 'degree_target']

        cfeatures0 = set()
        for s in selectors:
            cfeatures0 |= set(feat_selector[s])

        extra_features = [c for c in list(cfeatures0) if ('same' in c or 'eff' in c) and ('_im_ud' in c)]
        cfeatures_ += extra_features

    elif predict_mode_ == 'full':
        selectors = ['claim', 'batch']
        target_ = 'bdist'

        feat_version = 21
        excl_columns = ()

        col_families = generate_feature_groups(
            expanduser('~/data/kl/columns/v{0}_columns.txt'.format(feat_version)))

        feature_dict = deepcopy(col_families)
        feature_dict = {k: list(v) for k, v in feature_dict.items() if not any([c in v for c in excl_columns])}

        feature_dict_inv = {}

        for k, v in feature_dict.items():
            feature_dict_inv.update({x: k for x in v})

        excl_set = {'bdist_ma_None', 'bdist_ma_2'}

        cfeatures0 = set()
        for s in selectors:
            cfeatures0 |= set(feat_selector[s])

        gw_excl = [c for c in list(cfeatures0) if sum(df_dict['gw'][c].isnull()) > 0]
        lit_excl = [c for c in list(cfeatures0) if sum(df_dict['lit'][c].isnull()) > 0]

        cfeatures_ = list(cfeatures0 - (set(gw_excl) | set(lit_excl) | excl_set))
    else:
        cfeatures_ = None
        target_ = None

    return df_dict, cfeatures_, target_


def savefigs(report_, pred_mode, master_col, xlabel,
             fpath_figs=expanduser('~/data/kl/figs/')):

    print(f'mode: {pred_mode}')

    acc = []
    for depth, items in report_.items():
        for items2 in items:
            for item in items2:
                it, origin, j, dfs, clf, mdict, coeffs, cfeats = item
                fpr, tpr, _ = mdict['roc_curve']
                auc = integral_linear(fpr, tpr)
                acc.append((depth, origin, 'test', auc))
                fpr, tpr, _ = mdict['train_report']['roc_curve']
                auc2 = integral_linear(fpr, tpr)
                acc.append((depth, origin, 'train', auc2))

    df_acc = pd.DataFrame(acc, columns=[master_col, 'origin', 'sample', 'value'])

    for origin in ['gw', 'lit']:
        fig = plt.figure(figsize=(8, 8))
        rect = [0.15, 0.15, 0.75, 0.75]
        ax = fig.add_axes(rect)
        df_acc2 = df_acc.loc[df_acc['origin'] == origin]
        sns.lineplot(data=df_acc2, x=master_col, y='value', hue='sample', ax=ax)
        plt.xlabel(xlabel)
        plt.ylabel('AUC ROC value')
        fig.savefig(join(fpath_figs, f'{pred_mode}_{origin}_auc_{master_col}.pdf'))
        fig.savefig(join(fpath_figs, f'{pred_mode}_{origin}_auc_{master_col}.png'),
                    bbox_inches='tight', dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--mode',
                        default='posneg',
                        help='type of data to work with [gw, lit]')

    parser.add_argument('-n', '--niter',
                        type=int,
                        default=1,
                        help='test on the head of the dataset')

    parser.add_argument('-s', '--seed',
                        type=int,
                        default=13,
                        help='test on the head of the dataset')

    args = parser.parse_args()
    predict_mode = args.mode
    seed = args.seed
    n_iter = args.niter

    print(f'mode: {predict_mode}')

    df_dict, cfeatures, target = prepare_datasets(predict_mode)

    max_len_thr = 1
    forest_flag = True
    verbose = False
    if predict_mode == 'neutral':
        oversample = True
    else:
        oversample = False

    #***
    # depth
    complexity_dict = {'max_depth': 6, 'n_estimators': 100}
    min_samples_leaf_frac = 0.05
    depths = list(range(1, 7))
    sreport = {k: [] for k in depths}
    for cur_depth in depths:
        print(f'***{cur_depth}')
        complexity_dict['max_depth'] = cur_depth
        report = run_neut_models(df_dict, cfeatures,
                                 seed=seed,
                                 max_len_thr=max_len_thr,
                                 n_iter=n_iter,
                                 forest_flag=forest_flag, asym_flag=False,
                                 target=target,
                                 complexity_dict=complexity_dict,
                                 min_samples_leaf_frac=min_samples_leaf_frac,
                                 oversample=oversample,
                                 verbose=False)
        sreport[cur_depth].append(report)

    savefigs(sreport, predict_mode, 'depth', 'depth of decision tree')

    #***
    # min leaf size

    complexity_dict = {'max_depth': 1, 'n_estimators': 100}
    min_leaves = np.arange(0.005, 0.1, 0.005)
    sreport = {k: [] for k in min_leaves}

    for min_leaf in min_leaves:
        print(f'***{min_leaf}')
        report = run_neut_models(df_dict, cfeatures,
                                 seed=seed,
                                 max_len_thr=max_len_thr, n_iter=n_iter,
                                 forest_flag=forest_flag, asym_flag=False,
                                 target=target,
                                 complexity_dict=complexity_dict,
                                 min_samples_leaf_frac=min_leaf,
                                 oversample=oversample,
                                 verbose=False)
        sreport[min_leaf].append(report)

    savefigs(sreport, predict_mode, 'leaf_frac',
             'min leaf size as fraction dataset size')

    #***
    # n estimators

    complexity_dict = {'max_depth': 1, 'n_estimators': 100}
    min_leaf = 0.04
    estimators = np.arange(10, 200, 10)
    sreport = {k: [] for k in estimators}

    for x in estimators:
        print(f'***{x}')
        complexity_dict['n_estimators'] = x
        report = run_neut_models(df_dict, cfeatures,
                                 seed=seed,
                                 max_len_thr=max_len_thr,
                                 n_iter=n_iter,
                                 forest_flag=forest_flag, asym_flag=False,
                                 target=target,
                                 complexity_dict=complexity_dict,
                                 min_samples_leaf_frac=min_leaf,
                                 oversample=oversample,
                                 verbose=False)
        sreport[x].append(report)

    savefigs(sreport, predict_mode, 'n_estimators', 'number of estimators')