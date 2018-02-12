from itertools import product
from os.path import join, expanduser
import pandas as pd
import numpy as np
import pickle
import gzip
import matplotlib.pyplot as plt
import statsmodels.api as sm
from numpy import histogram, argmin, flatnonzero
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from bm_support.reporting import get_id_up_dn_df, get_lincs_df
from datahelpers.constants import iden, ye, ai, ps, up, dn, ar, ni, cexp, qcexp, nw, wi
from datahelpers.dftools import dict_to_array, accumulate_dicts
from sklearn.cluster import KMeans
from .gap_stat import choose_nc


def get_dataset(fpath_batches, origin, version, datatype, batchsize, cutoff_len, a, b,
                hash_int=None, **kwargs):
    if hash_int:
        fname = 'data_batches_{0}_v_{1}_hash_{2}.pgz'.format(origin, version, hash_int)
    else:
        fname = 'data_batches_{0}_v_{1}_c_{2}_m_{3}_n_{4}_a_{5}_b_{6}.pgz'.format(origin, version, datatype,
                                                                                  batchsize, cutoff_len, a, b)

    with gzip.open(join(fpath_batches, fname)) as fp:
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


def generate_samples(origin, version, lo, hi, n_batches, cutoff_len,
                     data_columns=[ye, iden, ai, ar, ps], complete_agg=True, verbose=False, hash_int=None):
    o_columns = [up, dn]

    data_cols = '_'.join(data_columns)

    batchsize = n_batches

    keys = ('version', 'cutoff_len')
    values = (version, cutoff_len)
    batches_path = expanduser('~/data/kl/batches')

    invariant_args = {
        'origin': origin,
        'datatype': data_cols,
        'batchsize': batchsize,
        'a': lo,
        'b': hi,
        'fpath': expanduser('~/data/kl/claims'),
        'fpath_batches': batches_path,
        'hash_int': hash_int
    }

    larg = {k: v for k, v in zip(keys, values)}
    print(larg)
    full_arg = {**invariant_args, **larg}
    print(full_arg)
    df_stats = get_id_up_dn_df(**full_arg)

    # list of dicts of numpy arrays
    dataset = get_dataset(**full_arg)
    dr = accumulate_dicts(dataset)
    arr2 = dict_to_array(dr)
    if ni in data_columns:
        ind = data_columns.index(ni)
        data_columns.pop(ind)
        arr2 = np.delete(arr2, ind, axis=0)
    print(arr2.shape)
    df_claims = pd.DataFrame(arr2.T, columns=([ni] + data_columns))
    df_claims[ni] = df_claims[ni].astype(int)

    # experimental
    df_exp = get_lincs_df(**full_arg)
    # df_exp['cdf'] = df_exp['score'].apply(lambda x: norm.cdf(x))
    m1 = (df_exp['pert_type'] == 'trt_oe')
    m2 = (df_exp['pert_itime'] == '96 h')
    m3 = (df_exp['is_touchstone'] == 1)
    m4 = (df_exp['pert_idose'] == '1 µL') | (df_exp['pert_idose'] == '2 µL')

    df_exp_cut = df_exp[m1 & m2 & m3 & m4]
    if complete_agg:
        agg_columns = [up, dn]
    else:
        agg_columns = [up, dn, 'pert_type', 'cell_id', 'pert_idose', 'pert_itime', 'is_touchstone']
    dfe = df_exp_cut.groupby(agg_columns).apply(lambda x:
                                                pd.Series([np.mean(x['score']), np.std(x['score'])],
                                                          index=['mean', 'std'])).reset_index()
    dfe[cexp] = dfe['mean'].apply(lambda x: norm.cdf(x))

    dfe = dfe[o_columns + [cexp, 'std']]
    dfe2 = pd.merge(dfe, df_stats.reset_index(), on=o_columns, how='left')
    dft = pd.merge(dfe2, df_claims, on=ni, how='inner')
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
                    thresholds=(-1.e-8, 0.5, 1.0),
                    verbose=False):
    if verbose:
        print(thresholds)
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
               exp_column='cdf_exp', qexp_column=qcexp,
               thresholds=(-1.e-8, 0.5, 1.0),
               distance_column='guess'):

    df = define_distance(df, exp_column, distance_column, qexp_column, thresholds, verbose)

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
    logreg = LogisticRegression(C=1./regularizer, tol=1e-6, penalty='l1', fit_intercept=True,
                                random_state=seed, warm_start=True)
    logreg = logreg.fit(X_train, y_train)

    rep = dict(zip(covariate_columns, logreg.coef_[0]))
    rep['intercept'] = logreg.intercept_[0]
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

    if fname or show:
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
    if n_classes > 1:
        km = KMeans(n_classes, tol=tol, random_state=seed)
        args = km.fit_predict(data)
        unis = np.unique(args)
        if n_classes > len(unis):
            print('Warning {0} classes predict {1} unique membres'.format(n_classes, len(np.unique(args))))
        centers = np.concatenate([km.cluster_centers_[k] for k in unis])
        if centers.ndim == 1:
            centers = centers.reshape(-1, 1)
        # print(centers, type(centers), centers.shape)
        mu_order = np.argsort(list(map(lambda x: x[0], centers)))
        mus = centers[mu_order]

        aligned_args = np.array([mu_order[a] for a in args])

        stds = [np.std(data[aligned_args == k], axis=0) for k in range(len(unis))]

        # replace zeros in stds (if all datapoints from a class are a constant)
        stds = [replace_zeros(s) for s in stds]

        # tensor of closest mus
        local_mu = np.stack([mus[j] for j in args])
        local_stds = np.stack([stds[j] for j in args])
        dists = np.true_divide(data-local_mu, local_stds, where=(local_stds != 0))
        acc = np.stack([np.repeat(len(unis), data.shape[0]), aligned_args], axis=1)
        acc = np.append(acc, dists, axis=1)
    else:
        std = data.std(axis=0)
        data_rel = np.true_divide((data - data.mean(axis=0)), std, where=(std != 0))
        r = np.repeat(np.array([1, 0])[:, None], data.shape[0], axis=1).T
        acc = np.hstack([r, data_rel])
    return acc


def cluster_optimally_(data, nc_max=2, override_negative=False):
    nc = choose_nc(data, nc_max)
    if override_negative and nc == -1:
        nc = 2
    if nc > 0:
        r = identify_intracluster_distances(data, nc)
    else:
        r = None
    return r


def cluster_optimally_pd(data, nc_max=2, min_size=5):
    # cluster optimally pandas
    data_ = data.values

    # make a matrix
    if data_.ndim == 1:
        data_ = data_.reshape(-1, 1)

    if data_.shape[0] < min_size:
        r = identify_intracluster_distances(data_, 1)
    else:
        r = cluster_optimally_(data_, nc_max, True)
    columns = [nw, wi] + ['d{0}'.format(j) for j in range(r.shape[1]-2)]
    return pd.DataFrame(r, index=data.index, columns=columns)


def groupby_normalize(data):
    data_ = data.values
    # make a matrix
    if data_.ndim == 1:
        data_ = data_.reshape(-1, 1)

    std = data_.std(axis=0)
    data_rel = np.true_divide((data_ - data_.mean(axis=0)), std, where=(std != 0))
    columns = ['{0}'.format(j) for j in range(data_rel.shape[1])]
    return pd.DataFrame(data_rel, index=data.index, columns=columns)


# def optimal_2split(data, verbose=False):
#     """
#     data 1d numpy array
#     """
#     cnts, bbs = histogram(data)
#     if verbose:
#         print(cnts, bbs)
#     diff = cnts[1:] - cnts[:-1]
#     derivative_change = diff[1:]*diff[:-1]/np.abs(diff[1:]*diff[:-1])
#     # sign change indices
#     ii = flatnonzero(derivative_change == -1)
#     arg_glo_min = argmin(cnts[1+ii])
#     lbbs, rbbs = bbs[1+ii[arg_glo_min]], bbs[2+ii[arg_glo_min]]
#     optimal_split = 0.5*(lbbs + rbbs)
#     return optimal_split


def optimal_2split(data, dicrete=True, equidistant=True, verbose=False):
    """
    data 1d numpy array
    """
    if dicrete and equidistant:
        uniqs = np.unique(data)
        uniqs = np.sort(uniqs)
        delta_ = (uniqs[1:] - uniqs[:-1]).min()
        rho_crit = 10
        n0 = data.shape[0]
        l = uniqs.max() - uniqs.min()
        rho_cur = n0/(l/delta_)
        delta = np.ceil(rho_crit/rho_cur)*delta_
        bbs = np.arange(uniqs.min() - 0.5*delta, uniqs.max() + 0.5*delta + 1e-6*delta, delta)
        # n_actual = (bbs[-1] - bbs[0])/delta
        if verbose:
            print(bbs)
    else:
        bbs = 10
    cnts, bbs = histogram(data, bbs)
    diff = cnts[1:] - cnts[:-1]
    ddif = diff[1:] - diff[:-1]
    if verbose:
        print(cnts, bbs)
        print(list(zip(range(len(cnts)), cnts)), bbs)
        print('f prime:', diff)
        print('f double prime:', ddif)
    # either diff == 0, or
    derivative_change = diff[1:]*diff[:-1]
    # sign change indices
    ii = flatnonzero((derivative_change < 0) & (ddif > 0))
    jj = flatnonzero(diff == 0)
    concats = np.concatenate([ii, jj])
    if verbose:
        print('candidate indices')
        print(1+ii, 1+jj)
        print(concats)
        print('candidate cnts')
        print(cnts[1+concats])
    if concats.size > 0:
        arg_glo_min = argmin(cnts[1+concats])
        if arg_glo_min < len(ii):
            lbbs, rbbs = bbs[1 + concats[arg_glo_min]], bbs[2 + concats[arg_glo_min]]
            optimal_split = 0.5*(lbbs + rbbs)
        else:
            # two conseq. equal values, optinal split is between them
            optimal_split = bbs[2 + concats[arg_glo_min]]
        if verbose:
            print(arg_glo_min, 1 + concats[arg_glo_min], cnts[1 + concats[arg_glo_min]], optimal_split)
    else:
        return np.nan
    return optimal_split


def optimal_2split_pd(data):
    data_ = data.values
    if data.unique().shape[0] > 1:
        split = optimal_2split(data)
    else:
        split = np.nan
    if np.isnan(split):
        m = data_.mean(axis=0)
        std = data_.std(axis=0)
        data_rel = np.array(data_, dtype=float)
        data_rel = data_rel - m
        if (std != 0) & ~np.isnan(std):
            data_rel /= std
        data_nw = np.ones(shape=data_.shape)
        data_wi = np.zeros(shape=data_.shape)
    else:
        ii1 = np.flatnonzero(data_ <= split)
        ii2 = np.flatnonzero(data_ > split)
        m1, m2 = data_[ii1].mean(), data_[ii2].mean()
        std1, std2 = data_[ii1].std(), data_[ii2].std()
        data_rel = np.array(data_, dtype=float)
        data_rel[ii1] = data_[ii1] - m1
        if (std1 != 0) & ~np.isnan(std1):
            data_rel[ii1] /= std1
        data_rel[ii2] = data_[ii2] - m2
        if (std2 != 0) & ~np.isnan(std2):
            data_rel[ii2] /= std2
        data_nw = 2*np.ones(shape=data_.shape)
        data_wi = np.zeros(shape=data_.shape)
        data_wi[ii2] = 1.0
    acc = np.vstack([data_nw, data_wi, data_rel])
    columns = [nw, wi] + ['d0']
    df_ = pd.DataFrame(acc.T, index=data.index, columns=columns)
    return df_
