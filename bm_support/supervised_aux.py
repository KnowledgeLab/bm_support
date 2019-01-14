from numpy.random import RandomState
from sklearn.model_selection import train_test_split
from .supervised import simple_stratify
from .supervised import problem_type_dict
from .supervised import select_features_dict, logit_pvalue, linear_pvalue, report_metrics
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
import pandas as pd
import matplotlib.pyplot as plt
from datahelpers.constants import up, dn


metric_selector = dict(zip(['corr', 'accuracy', 'precision', 'recall', 'f1'], range(5)))


def split_three_way(dfw, seed, target):
    rns = RandomState(seed)
    if len(dfw[target].unique()) < 5:
        strat = dfw[target]
    else:
        strat = None
    df_train, df_testgen = train_test_split(dfw, test_size=0.4,
                                            random_state=rns, stratify=strat)

    df_valid, df_test = train_test_split(df_testgen, test_size=0.5,
                                         random_state=rns)
    return df_train, df_test, df_valid


def study_sample(seed, dfw, target, feature_dict,
                 metric_mode, model_type, n_subtrials, n_estimators,
                 log_reg_dict={'min_log_alpha': -2., 'max_log_alpha': 2.},
                 verbose=False):
    nmax = 10000
    metric_uniform_exponent = 0.5
    mode_scores = None
    min_log_alpha, max_log_alpha = log_reg_dict['min_log_alpha'], log_reg_dict['max_log_alpha']

    feature_dict_inv = {}

    for k, v in feature_dict.items():
        feature_dict_inv.update({x: k for x in v})

    df_train, df_test, df_valid = split_three_way(dfw, seed, target)

    vc = df_train[target].value_counts()

    if verbose and vc.shape[0] < 5:
        print('*** df_train dist vc')
        print(vc)

    if len(dfw[target].unique()) < 5:
        # training on the normalized frequencies
        df_train2 = simple_stratify(df_train, target, seed, ratios=(2, 1, 1))
    else:
        df_train2 = df_train
    if model_type == 'rf' or model_type == 'rfr':
        param_dict = {'n_estimators': n_estimators, 'max_features': None, 'n_jobs': 1}
    else:
        param_dict = {'n_jobs': 1}

    meta_agg = []
    models = []

    rns = RandomState(seed)

    if model_type == 'rf' or model_type == 'rfr':
        enums = rns.randint(nmax, size=n_subtrials)
    elif model_type == 'lr':
        delta = (max_log_alpha - min_log_alpha) / n_subtrials
        enums = 1e1 ** np.arange(min_log_alpha, max_log_alpha, delta)
    else:
        enums = [1]

    for ii in enums:
        if model_type == 'rf' or model_type == 'rfr':
            param_dict['random_state'] = ii
        elif model_type == 'lr' or model_type == 'la':
            param_dict['C'] = ii

        # for random forest different seed yield different models, for logreg models are penalty-dependent
        cfeatures, chosen_metrics, test_metrics, model_ = select_features_dict(df_train2, df_test, target,
                                                                               feature_dict,
                                                                               model_type=model_type,
                                                                               max_features_consider=8,
                                                                               metric_mode=metric_mode,
                                                                               mode_scores=mode_scores,
                                                                               metric_uniform_exponent=metric_uniform_exponent,
                                                                               model_dict=param_dict,
                                                                               verbose=verbose)

        rmetrics = report_metrics(model_, df_valid[cfeatures], df_valid[target],
                                  mode_scores=mode_scores,
                                  metric_uniform_exponent=metric_uniform_exponent,
                                  metric_mode=metric_mode, problem_type=problem_type_dict[model_type])

        ii_dict = dict()
        ii_dict['run_par'] = seed
        ii_dict['current_features'] = cfeatures
        ii_dict['current_metrics'] = chosen_metrics
        ii_dict['test_metrics'] = test_metrics
        ii_dict['validation_metrics'] = rmetrics
        ii_dict['model'] = model_
        ii_dict['corr_all'] = dfw[cfeatures + [target]].corr()[target]
        ii_dict['corr_all_test'] = df_test[cfeatures + [target]].corr()[target]
        ii_dict['corr_all_valid'] = df_valid[cfeatures + [target]].corr()[target]

        if model_type[0] == 'l':
            if model_type == 'lr':
                ii_dict['pval_errors'] = logit_pvalue(model_, df_train2[cfeatures])
            elif model_type == 'lrg':
                ii_dict['pval_errors'] = linear_pvalue(model_, df_train2[cfeatures], df_train2[target])

        meta_agg.append(ii_dict)
        models.append(model_)

    validation_metrics_vec = [x['validation_metrics'] for x in meta_agg]
    test_metrics_vec = [x['test_metrics'] for x in meta_agg]

    main_metric_vec = [x['main_metric'] for x in validation_metrics_vec]

    index_best_run = np.argmax(main_metric_vec)
    best_features = meta_agg[index_best_run]['current_features']
    best_feature_groups = [feature_dict_inv[f] for f in best_features]

    report_dict = dict()
    report_dict['seeds'] = enums
    report_dict['index_best_run'] = index_best_run
    report_dict['best_features'] = best_features
    report_dict['best_feature_groups'] = best_feature_groups
    report_dict['max_scalar_mm'] = np.max(main_metric_vec)
    report_dict['validation_metrics'] = validation_metrics_vec
    report_dict['test_metrics'] = test_metrics_vec
    report_dict['best_validation_metrics'] = validation_metrics_vec[index_best_run]
    report_dict['best_test_metrics'] = test_metrics_vec[index_best_run]
    report_dict['corr_all'] = dfw[best_features + [target]].corr()[target]
    report_dict['corr_all_test'] = df_test[best_features + [target]].corr()[target]
    report_dict['corr_all_valid'] = df_valid[best_features + [target]].corr()[target]
    report_dict['best_model'] = models[index_best_run]
    if model_type[0] == 'l':
        report_dict['pval_errors'] = meta_agg[index_best_run]['pval_errors']

    return report_dict, meta_agg


def find_optimal_model(X_train, y_train, max_features=None, verbose=False):
    def foo(x):
        clf = LogisticRegression('l1', C=x)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_train)
        acc = precision_score(y_train, y_pred,  pos_label=0)
        nzero = X_train.shape[1] - np.sum(clf.coef_ == 0., axis=1)
        # if acc > 1e-2:
        if verbose:
            print('c: {0:.3f}, acc: {1:.4f}, non zero coeffs: {2}'.format(x, acc, nzero))
        return acc, clf
    penalties = [10**p for p in np.arange(-2, 2.1, 0.1)]
    grid = [foo(p) for p in penalties]
    accs = [x for x, _ in grid]
    if max_features:
        clfs = [x for _, x in grid]
        co_nzero_lens = [sum(clf.coef_[0] != 0) for clf in clfs]
        co_len_condition = [True if x <= max_features else False for x in co_nzero_lens]
        accs = [acc for acc, flag in zip(accs, co_len_condition) if flag]
        clfs = [clf for clf, flag in zip(clfs, co_len_condition) if flag]
    if accs:
        ii = np.argmax(accs)
        clf = clfs[ii]
        return clf, penalties[ii], accs[ii]
    else:
        return None, 0, 0.


def produce_topk_model(clf, dft, features, target, plot_title=None, verbose=False):
    dft = dft.copy()
    y_test = dft[target]
    if verbose:
        print('Freq of 1s: {0}'.format(sum(dft[target])/dft.shape[0]))
    p_pos = clf.predict_proba(dft[features])[:, 0]
    dft['proba_pos'] = pd.Series(p_pos, index=dft.index)
    p_level_base = 1. - dft[target].sum()/dft.shape[0]
    top_ps = np.arange(0.02, 3*p_level_base, 0.02)
    precs = []
    for p_level in top_ps:
        p_thr = np.percentile(p_pos, 100*(1-p_level))
        y_pred = 1. - (dft['proba_pos'] > p_thr).astype(int)
        precs.append(precision_score(y_test, y_pred,  pos_label=0))
    baseline = 1.0 - sum(dft[target])/dft.shape[0]
    if plot_title:
        plt.plot(top_ps, precs)
        plt.axhline(y=baseline, color='r')
        plt.ylim([0, np.round(np.max(precs), 1) + 0.3])
        plt.title(plot_title)
    y_pred = clf.predict(dft[features])
    p_norm = precision_score(y_test, y_pred, pos_label=0)
    return top_ps, precs, p_norm
