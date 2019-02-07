from numpy.random import RandomState
from sklearn.model_selection import train_test_split
from .supervised import simple_stratify
from .supervised import problem_type_dict
from .supervised import select_features_dict, logit_pvalue, linear_pvalue, report_metrics
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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


def bisect_zero(f, a, b, eps=1e-6):
    fa, fb = f(a), f(b)
    if fa*fb < 0:
        while fa*fb < 0 and b - a > eps:
            m = 0.5*(a + b)
            fm = f(m)
            if fa*fm < 0:
                b, fb = m, fm
            else:
                a, fa = m, fm
        return m
    else:
        return np.nan


def sect_min(f, a, b, tol=1e-6):
    phi = 0.5*(1. + 5**0.5)
    c, d = b - (b - a) / phi, a + (b - a) / phi
    while abs(c-d) > tol:
        if f(c) < f(d):
            b = d
        else:
            a = c
        c, d = b - (b - a) / phi, a + (b - a) / phi
    return 0.5*(a + b)


def find_optimal_model(X_train, y_train, max_features=None, model_type=LogisticRegression,
                       kwargs={'penalty': 'l1', 'solver': 'liblinear', 'max_iter': 100},
                       penalty_name='C',
                       verbose=False):

    def foo(x):
        clf_ = model_type(**{**kwargs, **{penalty_name: x}})
        clf_.fit(X_train, y_train)
        y_pred = clf_.predict(X_train)
        if model_type == LogisticRegression:
            acc_ = precision_score(y_train, y_pred, pos_label=0)
            nzero = X_train.shape[1] - np.sum(clf_.coef_ == 0., axis=1)
            if verbose:
                print('c: {0:.3f}, acc: {1:.4f}, non zero coeffs: {2}'.format(x, acc_, nzero))
        else:
            acc_ = r2_score(y_train, y_pred)
            if verbose:
                print('c: {0:.3f}, acc: {1:.4f}'.format(x, acc_))

        return acc_, clf_

    if max_features:
        best_penalty = bisect_zero(lambda x: np.sum(foo(x)[1].coef_ != 0.) - max_features, 1e-3, 1e3)
    else:
        best_penalty = sect_min(lambda x: -foo(x)[0], 1e-2, 1e2)

    acc, clf = foo(best_penalty)
    return clf, best_penalty, acc


def produce_topk_model(clf, dft, features, target, verbose=False):
    dft = dft.copy()
    y_test = dft[target]
    if verbose:
        print('Freq of 1s: {0}'.format(sum(dft[target])/dft.shape[0]))
    p_pos = clf.predict_proba(dft[features])[:, 0]
    # p_pos = clf.predict_proba(dft[features])[:, 1]
    # print(clf.predict_proba(dft[features]).shape)
    # print(clf.predict_proba(dft[features])[:5, :])
    dft['proba_pos'] = pd.Series(p_pos, index=dft.index)
    # p_level_base = 1. - dft[target].sum()/dft.shape[0]
    top_ps = np.arange(0.01, 1.0, 0.01)
    precs = []
    recs = []
    for p_level in top_ps:
        p_thr = np.percentile(p_pos, 100*(1-p_level))
        y_pred = 1. - (dft['proba_pos'] >= p_thr).astype(int)
        # y_pred = (dft['proba_pos'] > p_thr).astype(int)
        # print(p_level, p_thr, sum(1. - y_pred))
        precs.append(precision_score(y_test, y_pred, pos_label=0))
        recs.append(recall_score(y_test, y_pred, pos_label=0))
        # precs.append(precision_score(y_test, y_pred))
        # recs.append(recall_score(y_test, y_pred))

    metrics_dict = {'level': top_ps, 'prec': precs, 'rec': recs}

    base_prec = 1.0 - sum(dft[target])/dft.shape[0]
    # base_prec = sum(dft[target])/dft.shape[0]

    metrics_dict['prec_base'] = base_prec
    if isinstance(clf, LogisticRegression):
        metrics_dict['c_opt'] = clf.C

    y_pred = clf.predict(dft[features])
    y_prob = clf.predict_proba(dft[features])[:, 0]
    # y_prob = clf.predict_proba(dft[features])[:, 1]

    metrics_dict['prec0'] = precision_score(y_test, y_pred, pos_label=0)
    metrics_dict['rec0'] = recall_score(y_test, y_pred, pos_label=0)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label=0)

    # metrics_dict['prec0'] = precision_score(y_test, y_pred)
    # metrics_dict['rec0'] = recall_score(y_test, y_pred)
    # fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    metrics_dict['roc_curve'] = fpr, tpr, thresholds

    metrics_dict['auc'] = roc_auc_score(y_test, 1.-y_prob)
    # metrics_dict['auc'] = roc_auc_score(y_test, y_prob)

    return metrics_dict


def plot_prec_recall(metrics_dict, ax=None, title=None, fname=None):

    ax_init = ax
    sns.set_style('whitegrid')
    alpha_level = 0.8
    linewidth = 0.6

    if not ax:
        fig = plt.figure(figsize=(6, 6))
        rect = [0.15, 0.15, 0.75, 0.75]
        ax = fig.add_axes(rect)
    if title:
        ax.set_title(title)

    lines = []
    if 'prec' in metrics_dict.keys() and 'level' in metrics_dict.keys():
        xcoords = metrics_dict['level']
        precs = metrics_dict['prec']
        pline = ax.plot(xcoords, precs, color='b', linewidth=linewidth,
                        alpha=alpha_level, label='precision')
        lines.append(pline)

    if 'prec_base' in metrics_dict.keys():
        ax.axhline(y=metrics_dict['prec_base'], color='b', linewidth=linewidth,
                   alpha=alpha_level)

    if 'rec' in metrics_dict.keys() and 'level' in metrics_dict.keys():
        xcoords = metrics_dict['level']
        recs = metrics_dict['rec']
        rline = ax.plot(xcoords, recs, color='r', linewidth=linewidth,
                        alpha=alpha_level, label='recall')
        lines.append(rline)
    ax.set_ylim(ymin=0)
    if not ax_init:
        ax.legend()
    if fname:
        plt.savefig(fname)
    return ax


def plot_auc(metrics_dict, ax=None, title=None, fname=None):

    ax_init = ax
    sns.set_style('whitegrid')
    alpha_level = 0.8
    linewidth = 0.6

    if not ax:
        fig = plt.figure(figsize=(6, 6))
        rect = [0.15, 0.15, 0.75, 0.75]
        ax = fig.add_axes(rect)
    if title:
        ax.set_title(title)

    if 'roc_curve' in metrics_dict.keys():
        fpr, tpr, thresholds = metrics_dict['roc_curve']
        ax.plot(fpr, tpr,  linewidth=2*linewidth, alpha=alpha_level,
                label='AUC={0:.3f}'.format(metrics_dict['auc']))
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2*linewidth)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC')

    if not ax_init:
        ax.legend(loc="lower right")

    if fname:
        plt.savefig(fname)

    return ax


def level_corr(df_, groupby_cols, target_col, average_col, step=0.02, percentile_flag=True,
               ax=None, title=None, fname=None, verbose=False):
    df = df_.copy()
    ax_init = ax
    sns.set_style('whitegrid')
    alpha_level = 0.8
    linewidth = 0.6
    if verbose:
        print('start: {0}'.format(step))

    target_vec = df.groupby(groupby_cols).apply(lambda x: x[target_col].iloc[0])

    levels = np.arange(step, 1.0 + step, step)
    corrs = []
    corrs_agg = []
    distances_disc = []
    n_claims = []
    n_ints = []

    for l in levels:
        if verbose:
            print('level: {0}'.format(l))
        half_l = 0.5*l
        if percentile_flag:
            p_low = np.percentile(target_vec, 100*half_l)
            p_hi = np.percentile(target_vec, 100*(1.0 - half_l))
        else:
            p_low = half_l
            p_hi = 1.0 - half_l
        mask_low = (df[target_col] <= p_low)
        mask_hi = (df[target_col] >= p_hi)
        mask = mask_low | mask_hi
        df[target_col + 'b'] = 0.0
        df.loc[mask_hi, target_col + 'b'] = 1.0
        n_claims.append((l, sum(mask)))
        n_ints.append((l, df.loc[mask].drop_duplicates((up, dn)).shape[0]))

        if df.loc[mask].shape[0] > 2:
            c = df.loc[mask, [target_col, average_col]].corr().values[0, 1]
            corrs.append((l, c))

        dft = df.loc[mask].groupby(groupby_cols).apply(lambda x: pd.Series([x[target_col].iloc[0],
                                                                           x[average_col].mean()]))
        if dft.shape[0] > 2:
            corrs_agg.append((l, dft.corr().values[0, 1]))

        if df.loc[mask].shape[0] > 2:
            df['d'] = (df[target_col + 'b'] - df[average_col]).abs()
            dist_score = df.loc[mask, 'd'].mean()
            distances_disc.append((l, dist_score))

    if not ax:
        fig = plt.figure(figsize=(6, 6))
        rect = [0.15, 0.15, 0.75, 0.75]
        ax = fig.add_axes(rect)
    if title:
        ax.set_title(title)

    dd = {target_col: r'$\pi_\alpha$',
          average_col: r'$y_{i\alpha}$'
          }
    x1, y1 = np.array(corrs).T
    pline = ax.plot(x1, y1, color='r', linewidth=linewidth,
                    alpha=alpha_level, label=r'corr. ' + dd[target_col] + ' ' +  dd[average_col])

    # x2, y2 = np.array(corrs_agg).T
    # pline2 = ax.plot(x2, y2, color='r', linewidth=linewidth,
    #                  alpha=alpha_level, label='corr {0} {1} aggr.'.format(dd[target_col], dd[average_col]))

    x3, y3 = np.array(distances_disc).T
    pline3 = ax.plot(x3, y3, color='g', linewidth=linewidth,
                     alpha=alpha_level, label=r'mean $b_i\alpha$')
    ax2 = ax.twinx()
    # x4, y4 = np.array(n_claims).T
    x4, y4 = np.array(n_ints).T
    pline4 = ax2.plot(x4, y4, label='number of interactions')
    ax2.set_yscale("log", nonposy='clip')

    # lns = pline + pline2 + pline3 + pline4
    lns = pline + pline3 + pline4
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='center right')
    ax.set_ylim(ymin=0.0)
    if fname:
        plt.savefig(fname)
    return ax


def level_corr_one_tail(df_, groupby_cols, target_col, corr_cols, step=0.02,
                        direct_sweep_order=True,
                        ax=None, title=None, verbose=False):
    df = df_.copy()
    sns.set_style('whitegrid')
    alpha_level = 0.8
    linewidth = 0.6

    target_vec = df.groupby(groupby_cols).apply(lambda x: x[target_col].iloc[0]).values
    if verbose:
        print(target_vec[:5])

    levels = np.arange(0.0, 1.0+step, step)
    corrs = []
    n_claims = []
    for level in levels:
        if direct_sweep_order:
            p = np.percentile(target_vec, 100*level)
            mask = (df[target_col] <= p)
        else:
            p = np.percentile(target_vec, 100*(1. - level))
            mask = (df[target_col] >= p)

        if sum(mask) > 2:
            c = df.loc[mask, corr_cols].corr().values[0, 1]
            corrs.append((level, c))

        if sum(mask) > 2:
            n_claims.append((level, sum(mask)))

    if not ax:
        fig = plt.figure(figsize=(6, 6))
        rect = [0.15, 0.15, 0.75, 0.75]
        ax = fig.add_axes(rect)
    if title:
        ax.set_title(title)

    lns = []
    x1, y1 = np.array(corrs).T
    pline = ax.plot(x1, y1, color='b', linewidth=linewidth,
                    alpha=alpha_level, label='corr {0} {1}'.format(target_col, corr_cols[0], corr_cols[1]))
    lns.append(pline)

    if n_claims:
        ax2 = ax.twinx()
        x4, y4 = np.array(n_claims).T
        pline4 = ax2.plot(x4, y4, label='number of interactions')
        ax2.set_yscale("log", nonposy='clip')
        lns.append(pline4)

    lns = pline + pline4
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs)
    ax.set_ylim(ymin=0.0)

    return ax

