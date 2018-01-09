from itertools import product
from os.path import join, expanduser
import pandas as pd
import numpy as np
import pickle
import gzip
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from bm_support.reporting import get_id_up_dn_df, get_lincs_df
from datahelpers.constants import iden, ye, ai, ps, up, dn, ar, ni, cexp, qcexp
from datahelpers.dftools import dict_to_array, accumulate_dicts
from sklearn.cluster import KMeans
from .gap_stat import choose_nc


def get_dataset(fpath_batches, origin, version, datatype, batchsize, cutoff_len, a, b, **kwargs):
    with gzip.open(join(fpath_batches,
                        'data_batches_{0}_v_{1}_c_{2}_m_{3}_n_{4}_a_{5}_b_{6}.pgz'.format(origin, version, datatype,
                                                                                          batchsize, cutoff_len, a, b)
                        )) as fp:
        dataset = pickle.load(fp)
        return dataset


def load_samples(origin, version, lo, hi, n_batches, cutoff_len):

    feauture_cols = [ai, ar]
    data_columns = [ye, iden] + feauture_cols + [ps]
    data_cols = '_'.join(data_columns)

    origin_cur = origin
    batchsize = n_batches
    versions = [version]

    cutoff_lens = [cutoff_len]
    keys = ('version', 'cutoff_len', 'case')

    batches_path = expanduser('~/data/kl/batches')

    invariant_args = {
        'origin': origin_cur,
        'datatype': data_cols,
        'batchsize': batchsize,
        'a': lo,
        'b': hi,
        'fpath': expanduser('~/data/kl/claims'),
        'fpath_batches': batches_path
    }

    largs = [{k: v for k, v in zip(keys, p)} for p in product(*(versions, cutoff_lens))]
    full_largs = [{**invariant_args, **dd} for dd in largs]
    print(full_largs[0])

    ids_list = []

    dfs = [get_id_up_dn_df(**dd) for dd in full_largs]
    ids_list.extend(list(zip(full_largs, dfs)))

    datasets = [get_dataset(**dd) for dd in full_largs]

    # pick literature df
    ds = datasets[0]
    dr = accumulate_dicts(ds)
    return dr


def generate_samples(origin, version, lo, hi, n_batches, cutoff_len, verbose=False):
    o_columns = [up, dn]

    feauture_cols = [ai, ar]
    data_columns = [ye, iden] + feauture_cols + [ps]
    data_cols = '_'.join(data_columns)

    origin_cur = origin
    batchsize = n_batches
    versions = [version]

    cutoff_lens = [cutoff_len]
    keys = ('version', 'cutoff_len', 'case')

    batches_path = expanduser('~/data/kl/batches')

    invariant_args = {
        'origin': origin_cur,
        'datatype': data_cols,
        'batchsize': batchsize,
        'a': lo,
        'b': hi,
        'fpath': expanduser('~/data/kl/claims'),
        'fpath_batches': batches_path
    }

    largs = [{k: v for k, v in zip(keys, p)} for p in product(*(versions, cutoff_lens))]
    full_largs = [{**invariant_args, **dd} for dd in largs]
    if verbose:
        print(full_largs[0])

    ids_list = []
    lincs_list = []

    dfs = [get_id_up_dn_df(**dd) for dd in full_largs]
    ids_list.extend(list(zip(full_largs, dfs)))

    datasets = [get_dataset(**dd) for dd in full_largs]

    # pick literature df
    ds = datasets[0]

    dr = accumulate_dicts(ds)
    arr2 = dict_to_array(dr)
    df = pd.DataFrame(arr2.T, columns=([ni] + data_columns))

    df[ni] = df[ni].astype(int)

    # experimental
    dfls = [get_lincs_df(**dd) for dd in full_largs]
    lincs_list.extend(list(zip(full_largs, dfls)))

    args_lincs_list = lincs_list

    # consider only the first report
    for args, dfl in args_lincs_list[0:1]:
        # find model run reports that align with current lincs item
        ccs = ['pert_itime', 'cell_id', 'pert_idose', 'pert_type', 'is_touchstone']
        cnts = [6, 5, 4, 4, 2]
        acc = []
        for c, i in zip(ccs, cnts):
            vc = dfl[c].value_counts()
            suffix = list(vc.iloc[:i].index)
            acc.append(suffix)

        dfl['cdf'] = dfl['score'].apply(lambda x: norm.cdf(x))

    args_lincs_std_list = []
    for args, dfl in args_lincs_list:
        dfl['cdf'] = dfl['score'].apply(lambda x: norm.cdf(x))
        m1 = (dfl['pert_type'] == 'trt_oe')
        m2 = (dfl['pert_itime'] == '96 h')
        m3 = (dfl['is_touchstone'] == 1)
        m4 = (dfl['pert_idose'] == '1 µL') | (dfl['pert_idose'] == '2 µL')

        dfc = dfl[m1 & m2 & m3 & m4]
        if verbose:
            print(args)
            print(dfc['cell_id'].value_counts())
            print(dfl.shape, dfc.shape)
        dfl_mean_std = dfc.groupby([up, dn, 'pert_type', 'cell_id',
                                    'pert_idose', 'pert_itime',
                                    'is_touchstone']).apply(lambda x: pd.Series([np.mean(x['cdf']), np.std(x['cdf'])],
                                                                                index=['mean', 'std'])).reset_index()
        args_lincs_std_list.append((args, dfl_mean_std))

    # pick lincs df
    dfl = args_lincs_std_list[0][1].copy()
    dfl = dfl[o_columns + ['mean']].rename(columns={'mean': 'cdf_exp'})
    if verbose:
        print('dfl null cexp: {0}'.format(sum(dfl[cexp].isnull())))

    # pick pairs df
    dfid = ids_list[0][1]

    dfexp = pd.merge(dfl, dfid.reset_index(), on=o_columns, how='left')
    if verbose:
        print('dfexp null cexp: {0}'.format(sum(dfexp[cexp].isnull())))

    dft = pd.merge(dfexp, df, on=ni, how='left')
    if verbose:
        print('dft null cexp: {0}'.format(sum(dft[cexp].isnull())))

    return dft


def stratify_df(df, column, size, frac, seed=17):
    """
    take a subsample from df that has frac fraction of the rarest value
    """
    replacement = False
    df_size = df.shape[0]

    vc = df[column].value_counts()
    rare_value = vc.index[-1]
    rare_size = vc.iloc[-1]
    mask = (df[column] == rare_value)
    ssample_rare_size = int(frac * size)
    if rare_size < ssample_rare_size:
        print('not enough rare values to produce frac*size ssample without replacement')
        replacement = True

    np.random.seed(seed)
    rare_inds = np.random.choice(rare_size, ssample_rare_size, replacement)
    rest_inds = np.random.choice(df_size - rare_size, size - ssample_rare_size, False)

    df_rare = df.loc[mask].iloc[rare_inds]
    df_rest = df.loc[~mask].iloc[rest_inds]
    dfr = pd.concat([df_rare, df_rest])
    return dfr


def smart_stratify_df(df, column, size=500, ratios=None, replacement=False, seed=17, verbose=False):
    vc = df[column].value_counts()
    if verbose:
        print(vc)
    if size > df.shape[0]:
        print('requested sample size is greater than available sample:')

    if isinstance(ratios, (list, tuple)) and len(ratios) == vc.shape[0]:
        sizes_list = np.array(ratios) / np.sum(ratios)
        sizes_list = [int(x * size) for x in sizes_list]
    else:
        f = 1. / vc.shape[0]
        sizes_list = [int(f * size)] * vc.shape[0]
    sizes_list[0] = size - np.sum(sizes_list[1:])
    if verbose:
        print('fracs: {0}'.format(sizes_list))
    masks = [(df[column] == v) for v in vc.index]
    replacements = [(x > y) or replacement for x, y in zip(sizes_list, vc)]
    if any(replacements):
        print('replacements sampling will be used')

    if replacements:
        print('replacements: {0}'.format(replacements))

    np.random.seed(seed)
    triplets = list(zip(sizes_list, vc, replacements))

    inds = [np.random.choice(msize, n, r) for n, msize, r in triplets]
    subdfs = [df.loc[m].iloc[ii] for m, ii in zip(masks, inds)]
    dfr = pd.concat(subdfs)
    return dfr


def normalize_columns(df, columns):
    df2 = df.copy()
    sc = MinMaxScaler()
    df2[columns] = sc.fit_transform(df[columns])
    return df2


def logreg_analysis(df, covariate_columns, stratify=False, statify_size=5000,
                    stratify_frac=0.5, regularizer=1.0, seed=17, fname=None, nfolds=3):

    print(df[ps].value_counts(), df[ps].mean())

    bexp = 'bool_exp'
    gu = 'guess'
    ma_pos = (df['cdf_exp'] > 0.5)
    print(sum(ma_pos))
    df[bexp] = 0.0
    df.loc[ma_pos, bexp] = 1.0
    print(df[bexp].value_counts(), df[bexp].mean())

    # prepare guess var
    mask_wrong_guess = (df[bexp] != df[ps])
    print(sum(mask_wrong_guess))
    df[gu] = 1.0
    df.loc[mask_wrong_guess, gu] = 0.0
    df[gu].value_counts(), df[gu].mean()

    for c in covariate_columns:
        print(c, df[c].min(), df[c].max())

    # prepare stratified sample
    if stratify:
        df2 = stratify_df(df, gu, statify_size, stratify_frac, seed)
    else:
        df2 = df

    X = df2[covariate_columns].values
    y = df2[gu].values

    logit_model = sm.Logit(y, X)
    result = logit_model.fit()
    print(result.summary())
    print(covariate_columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    logreg = LogisticRegression(C=1./regularizer, penalty='l1', fit_intercept=False)
    logreg = logreg.fit(X_train, y_train)
    print('### lr intercept and coefs')

    rep = list(zip(covariate_columns, list(map(lambda x: '{:.3f}'.format(x), logreg.coef_[0]))))
    print(rep)
    y_pred = logreg.predict(X_test)
    print('Accuracy of logistic regression classifier on test set: {:.5f}'.format(logreg.score(X_test, y_test)))

    kfold = model_selection.KFold(n_splits=nfolds, random_state=seed)
    modelCV = LogisticRegression(C=1./regularizer, penalty='l1', fit_intercept=False)
    scoring = 'accuracy'
    results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
    print('{0}-fold cross validation average accuracy: {1:.3f}'.format(nfolds, results.mean()))

    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)

    print(classification_report(y_test, y_pred))

    logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:, 1])
    fig = plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    fig.savefig(fname)
    plt.show()


def quantize_series(s1, thrs, verbose=False):
    s_out = s1.copy()
    lambda_ = len(thrs) - 1
    for a, b, ix in zip(thrs[:-1], thrs[1:], range(lambda_)):
        mask = (a < s1) & (s1 <= b)
        s_out[mask] = ix
        if verbose:
            print(a, b, ix, sum(mask))
    # return s_out / (lambda_ - 1)
    return s_out


def define_distance(df, exp_column=cexp, distance_column='guess', quantized_column=qcexp,
                    thresholds=[-1.e-8, 0.5, 1.0],
                    verbose=False):

    n_classes = len(thresholds) - 2
    if verbose:
        print(df[ps].value_counts())

    df[quantized_column] = quantize_series(df[exp_column], thresholds, verbose)

    if verbose:
        print(df[quantized_column].value_counts())

    df[distance_column] = np.abs(df[quantized_column] - df[ps]*n_classes)
    if verbose:
        print(df[distance_column].value_counts(), df[distance_column].mean())

    return df


def prepare_xy(df, covariate_columns, stratify=False, statify_size=5000,
               stratify_frac=0.5, seed=17, verbose=False,
               exp_column='cdf_exp', thresholds=[-1.e-8, 0.5, 1.0],
               distance_column='guess'):

    df = define_distance(df, exp_column, distance_column, thresholds, verbose)

    if verbose:
        for c in covariate_columns:
            print(c, df[c].min(), df[c].max())

    # prepare stratified sample
    if stratify:
        df2 = stratify_df(df, distance_column, statify_size, stratify_frac, seed)
    else:
        df2 = df

    X = df2[covariate_columns].values
    y = df2[distance_column].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

    return X_train, X_test, y_train, y_test


def rf_study(X_train, X_test, y_train, y_test, covariate_columns=[],
             seed=0, depth=None, fname=None, show=False, title_prefix=None):
    report = {}
    rf = RandomForestClassifier(max_depth=depth, random_state=seed)
    rf = rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    report['corr_train_pred'] = np.corrcoef(y_pred, y_test)[0, 1]
    conf_matrix = confusion_matrix(y_test, y_pred)
    report['confusion'] = conf_matrix

    report['class_report'] = classification_report(y_test, y_pred)

    positive_proba = rf.predict_proba(X_test)[:, 1]

    auroc = roc_auc_score(y_test, positive_proba)

    report['auroc'] = auroc
    importances = rf.feature_importances_
    stds = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)

    report['feature_importance'] = dict(zip(covariate_columns, importances))
    report['feature_importance_std'] = dict(zip(covariate_columns, stds))

    fpr, tpr, thresholds = roc_curve(y_test, positive_proba)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label='Random Forest: area = {0:.3f}'.format(report['auroc']))
    ax.plot([0, 1], [0, 1], 'r--')
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.axis('equal')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('{0} Random Forest ROC'.format(title_prefix))
    ax.legend(loc='lower right')

    if fname:
        fig.savefig(fname)
    if show:
        plt.show()
    else:
        plt.close()
    return report


def rf_study_multiclass(X_train, X_test, y_train, y_test, covariate_columns=[],
                        seed=0, depth=None, fname=None, show=False, title_prefix=None, n_estimators=20,
                        return_model=False):
    n_states = len(set(y_test))
    report = {}
    rf = RandomForestClassifier(max_depth=depth, random_state=seed, n_estimators=n_estimators)
    rf = rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    report['corr_train_pred'] = np.corrcoef(y_pred, y_test)[0, 1]
    conf_matrix = confusion_matrix(y_test, y_pred)
    report['confusion'] = conf_matrix

    report['class_report'] = classification_report(y_test, y_pred)

    positive_proba = rf.predict_proba(X_test)
    y_test_binary = label_binarize(y_test, classes=np.arange(0.0, n_states))

    aurocs = [roc_auc_score(y_, proba_) for proba_, y_ in zip(positive_proba.T, y_test_binary.T)]

    report['auroc'] = aurocs
    importances = rf.feature_importances_
    stds = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    report['feature_importance'] = dict(zip(covariate_columns, importances))
    report['feature_importance_std'] = dict(zip(covariate_columns, stds))

    coords = [roc_curve(y_, proba_) for proba_, y_ in zip(positive_proba.T, y_test_binary.T)]

    fig, ax = plt.subplots(figsize=(5, 5))
    for cs, auc, k in zip(coords, aurocs, range(n_states)):
        fpr, tpr, thresholds = cs
        distance = k/(n_states - 1)
        ax.plot(fpr, tpr, label='Random Forest: dist {0} area = {1:.3f}'.format(distance, auc))
    ax.plot([0, 1], [0, 1], 'r--')
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.axis('equal')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('{0} Random Forest ROC'.format(title_prefix))
    ax.legend(loc='lower right')

    if fname:
        fig.savefig(fname)
    if show:
        plt.show()
    else:
        plt.close()
    if return_model:
        return report, rf
    else:
        return report


def plot_importances(importances, stds, covariate_columns, fname, title_prefix, show=False):
    """
    importances, stds, covariate_columns are all lists of the same length
    :param importances:
    :param stds:
    :param covariate_columns:
    :param fname:
    :param title_prefix:
    :param show:
    :return:
    """

    if fname:
        indices = np.argsort(importances)[::-1]
        n = len(covariate_columns)

        imp_ccs = [covariate_columns[i] for i in indices]
        fig = plt.figure()
        plt.title('{0} Random Forest feature importances'.format(title_prefix))
        plt.bar(range(n), importances[indices],
                color='r', yerr=stds[indices], align='center', alpha=0.5)
        plt.xticks(range(n), imp_ccs)
        plt.xlim([-1, n])
        fig.savefig(fname)
        if show:
            plt.show()
        else:
            plt.close()


def plot_lr_coeffs_with_penalty(alphas, coeff_dict, covariate_names, fname=None, position='lower left',
                                logy=False, show=True, title_prefix=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    linestyles = ['-', '--', '-.', ':']
    lines = []
    arr = np.concatenate(list(coeff_dict.values()))

    for k in covariate_names:
        ls = linestyles[covariate_names.index(k) % len(linestyles)]
        if logy:
            y = np.abs(coeff_dict[k])
        else:
            y = coeff_dict[k]
        l = ax.plot(alphas, y, linewidth=1.5, ls=ls)
        lines.append(l[0])

    ax.set_xscale('log')
    ax.set_xlim([1e1 ** np.floor(np.log10(np.min(alphas))),
                 1e1 ** np.ceil(np.log10(np.max(alphas)))])

    if logy:
        ax.set_yscale('log')
        ax.set_ylim([1e1 ** np.floor(np.log10(np.min(np.abs(arr[np.nonzero(arr)])))),
                     1e1 ** np.ceil(np.log10(np.max(np.abs(arr))))])
    else:
        ax.set_ylim([np.floor(np.min(arr)), np.ceil(np.max(arr))])

    location = position
    ax.legend(lines, covariate_names, loc=location, frameon=True,
              framealpha=1.0, facecolor='w', edgecolor='k', shadow=False, prop={'size': 12})
    ax.set_title('{0} Logistic regression coeffs'.format(title_prefix))
    ax.set_xlabel('penalty')
    ax.set_ylabel('coefficient')

    if fname:
        fig.savefig(fname)
    if show:
        plt.show()
    else:
        plt.close()


def lr_study(X_train, X_test, y_train, y_test, covariate_columns=[], seed=0, regularizer=1.0,
             nfolds=3, fname=None, show=False, title_prefix=None):
    report = {}
    logreg = LogisticRegression(C=1./regularizer, tol=1e-6, penalty='l1', fit_intercept=False,
                                random_state=seed, warm_start=True)
    logreg = logreg.fit(X_train, y_train)

    rep = dict(zip(covariate_columns, logreg.coef_[0]))
    report['coeffs'] = rep

    y_pred = logreg.predict(X_test)
    report['corr_train_pred'] = np.corrcoef(y_pred, y_test)[0, 1]

    report['test_accuracy'] = logreg.score(X_test, y_test)
    positive_proba = logreg.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, positive_proba)
    report['auroc'] = roc_auc_score(y_test, positive_proba)

    kfold = model_selection.KFold(n_splits=nfolds, random_state=seed)
    scoring = 'accuracy'

    results = model_selection.cross_val_score(logreg, np.concatenate([X_train, X_test]),
                                              np.concatenate([y_train, y_test]),
                                              cv=kfold, scoring=scoring)

    report['{0}_fold_cv_ave_accuracy'.format(nfolds)] = results.mean()

    conf_matrix = confusion_matrix(y_test, y_pred)
    report['confusion'] = conf_matrix

    report['class_report'] = classification_report(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label='Logistic Regression: area = {0:.3f}'.format(report['auroc']))
    ax.plot([0, 1], [0, 1], 'r--')
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.axis('equal')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('{0} Log Reg ROC'.format(title_prefix))
    ax.legend(loc="lower right")

    if fname:
        fig.savefig(fname)
    if show:
        plt.show()
    else:
        plt.close()
    return report


def std_over_samples(lengths, means, stds):
    # https://stats.stackexchange.com/questions/43159/how-to-calculate-pooled-variance-of-two-groups-given-known-group-variances-mean
    # https://stats.stackexchange.com/questions/30495/how-to-combine-subsets-consisting-of-mean-variance-confidence-and-number-of-s
    total_length = np.sum(lengths)
    total_mean = np.sum(list(map(lambda x: x[0]*x[1], zip(lengths, means))))/total_length
    total_num = np.sum(list(map(lambda x: x[0]*(x[1]**2 + x[2]**2), zip(lengths, means, stds))))
    r = (total_num/total_length - total_mean**2)**0.5
    return r


def invert_bayes(df_input, clf, feature_columns, verbose=False):
    df = df_input.copy()
    arr_probs = clf.predict_proba(df[feature_columns])
    if verbose:
        print('names of feature columns:', feature_columns)
        print('probs shape: ', arr_probs.shape)
    arr_ps = df[ps].values
    p_min = np.float(5e-1/clf.n_estimators)
    arr_probs[arr_probs == 0.0] = p_min
    arr_probs2 = arr_probs/np.sum(arr_probs, axis=1).reshape(-1)[:, np.newaxis]

    tensor_claims = np.stack([1 - arr_ps, arr_ps])
    tensor_probs = np.stack([arr_probs2, arr_probs2[:, ::-1]])
    if verbose:
        print(tensor_claims.shape, tensor_probs.shape)
    tt = np.multiply(tensor_claims.T, tensor_probs.T)
    if verbose:
        print(tt.shape)
    tt = np.sum(tt, axis=2)
    if verbose:
        print(tt.shape)
    for v, j in zip(tt, range(tt.shape[0])):
        df['pe_{0}'.format(j)] = v
    return df


def transform_logp(x, barrier=20):
    y = x.copy()
    y[y > barrier] = barrier
    y[y < -barrier] = -barrier
    return y


def aggregate_over_claims(df, barrier):
    pes = ['pe_{0}'.format(j) for j in range(3)]
    p_agg = df.groupby(ni).apply(lambda x: np.sum(np.log(x[pes]), axis=0))
    p_agg2 = p_agg.apply(lambda x: x - sorted(x)[1], axis=1)
    p_agg3 = p_agg2.apply(lambda x: np.exp(transform_logp(x, barrier)), axis=1)
    p_agg4 = p_agg3.apply(lambda x: x/np.sum(x), axis=1)
    p_agg4 = p_agg4.merge(pd.DataFrame(df.drop_duplicates(ni)[[ni, 'qcdf_exp']]),
                          how='left', left_index=True, right_on=ni)
    return p_agg4


def kmeans_cluster(data, n_classes=2, seed=11, tol=1e-6, verbose=False, return_flags=True):
    # data shape npoints x ndim
    # init random state
    rns = np.random.RandomState(seed)
    mins = np.min(data, axis=0)
    maxs = np.max(data, axis=0)
    mus = [np.array([rns.uniform(x, y) for x, y in zip(mins, maxs)]) for k in range(n_classes)]
    delta = 1.
    while delta > tol:
        # calculate distances
        dists = [np.sum((data-mu)**2, axis=1)**0.5 for mu in mus]
        # calculate closest distances
        args = np.argmin(np.vstack(dists), axis=0)
        # update mus
        mus_new = [np.mean(data[args == k], axis=0) for k in range(n_classes)]
        mus = mus_new
    mus = sorted(mus, key=lambda y: y[0])
    dists = [np.sum((data-mu)**2, axis=1)**0.5 for mu in mus]
    # predicted classes
    args = np.argmin(np.vstack(dists), axis=0)
    if verbose:
        print('sum:', np.sum(args))
    stds = [np.std(data[args == k], axis=0) for k in range(n_classes)]

    def replace_zeros(x):
        x[x == 0] = 1.
        return x
    # replace zeros in stds (if all datapoints from a class are a constant)
    stds = [replace_zeros(s) for s in stds]

    if return_flags:
        # tensor of closest mus
        mu_tensor = np.dot(np.stack([1 - args, args]).T, np.stack(mus))
        # tensor of closest stds
        std_tensor = np.dot(np.stack([1 - args, args]).T, np.stack(stds))
        # vector of relative
        diffs = (data - mu_tensor)/std_tensor
#         return args, diffs
        return np.append(args.reshape((-1, 1)), diffs, axis=1)
    else:
        return mus, stds


def replace_zeros(x):
    x[x == 0] = 1.
    return x


def identify_intracluster_distances(data, n_classes=2, seed=11, tol=1e-6, verbose=False):
    km = KMeans(n_classes, tol=tol, random_state=seed)
    args = km.fit_predict(data)
    mus = sorted(km.cluster_centers_, key=lambda y: y[0])
    mu_order = np.argsort(list(map(lambda x: x[0], km.cluster_centers_)))
    stds = [np.std(data[args == k], axis=0) for k in range(n_classes)]
    # replace zeros in stds (if all datapoints from a class are a constant)
    stds = [replace_zeros(s) for s in stds]
    aligned_args = np.array([mu_order[a] for a in args])
    # tensor of closest mus
    local_mu = np.stack([km.cluster_centers_[j] for j in args])
    local_stds = np.stack([stds[j] for j in args])
    dists = (data - local_mu)/local_stds
    acc = np.stack([np.repeat(n_classes, data.shape[0]), aligned_args], axis=1)
    acc = np.append(acc, dists, axis=1)
    return acc


def cluster_optimally(data, nc_max=5):
    nc = choose_nc(data, nc_max)
    if nc > 0:
        r = identify_intracluster_distances(data, nc)
    else:
        r = None
    return r
