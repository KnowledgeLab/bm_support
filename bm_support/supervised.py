from itertools import product
from os.path import join, expanduser
import pandas as pd
import numpy as np
import pickle
import gzip
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from numpy import histogram, argmin, flatnonzero
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score,\
    f1_score, classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import label_binarize
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from bm_support.reporting import get_id_up_dn_df, get_lincs_df
from datahelpers.constants import iden, ye, ai, ps, up, dn, ar, ni, cexp, qcexp, nw, wi, dist, pm, ct, affs, aus
from datahelpers.dftools import select_appropriate_datapoints, dict_to_array, accumulate_dicts, add_column_from_file
from .add_features import prepare_final_df
from datahelpers.community_tools import get_community_fnames_cnames
from sklearn.cluster import KMeans
from scipy.stats import t as tdistr
from .gap_stat import choose_nc
from copy import deepcopy
from numpy.random import RandomState

problem_type_dict = {'rf': 'class', 'lr': 'class', 'rfr': 'regr', 'lrg': 'regr'}


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
                     data_columns=(ye, iden, ai, ar, ps), complete_agg=True, hash_int=None,
                     load_batches=False, lincs_type='', verbose=False):
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
        'hash_int': hash_int,
        'type': lincs_type
    }

    larg = {k: v for k, v in zip(keys, values)}
    full_arg = {**invariant_args, **larg}
    if verbose:
        print('larg: ', larg)
        print('full_arg: ', full_arg)
    df_stats = get_id_up_dn_df(**full_arg)

    if load_batches:
        # list of dicts of numpy arrays
        dataset = get_dataset(**full_arg)
        dr = accumulate_dicts(dataset)
        arr2 = dict_to_array(dr)
        if ni in data_columns:
            ind = data_columns.index(ni)
            data_columns.pop(ind)
            arr2 = np.delete(arr2, ind, axis=0)

        df_claims = pd.DataFrame(arr2.T, columns=([ni] + data_columns))
        df_claims[ni] = df_claims[ni].astype(int)
    else:
        if hash_int:
            fname = 'df_{0}_v_{1}_hash_{2}.h5'.format(origin, version, hash_int)
        else:
            fname = 'df_{0}_v_{1}_c_{2}_m_{3}_n_{4}_a_{5}_b_{6}.pgz'.format(origin, version, data_columns,
                                                                            batchsize, cutoff_len, lo, hi)
        if fname[-3:] == 'pgz':
            with gzip.open(join(batches_path, fname)) as fp:
                df_claims = pickle.load(fp)
        elif fname[-2:] == 'h5':
            df_claims = pd.read_hdf(expanduser(join(batches_path, fname)))

    if verbose:
        print('df size {0}, df unique [up, dn, pm] {1}'.format(df_claims.shape[0],
                                                               df_claims.drop_duplicates([up, dn, pm]).shape[0]))
        print('{0} dataframe size {1}'.format(origin, df_claims.shape[0]))
        print('unique statements {0}'.format(len(df_claims[ni].unique())))

    df_claims = df_claims.drop_duplicates([up, dn, pm])
    # experimental
    df_exp = get_lincs_df(**full_arg)

    if verbose:
        print('df_exp rows: {0}'.format(df_exp.shape[0]))
        print(df_exp.head())

    # df_exp['cdf'] = df_exp['score'].apply(lambda x: norm.cdf(x))
    m1 = (df_exp['pert_type'] == 'trt_oe')
    m2 = (df_exp['pert_itime'] == '96 h')
    m3 = (df_exp['is_touchstone'] == 1)
    m4 = (df_exp['pert_idose'] == '1 µL') | (df_exp['pert_idose'] == '2 µL')

    df_exp_cut = df_exp[m1 & m2 & m3 & m4]
    if verbose:
        print('df_exp_cut rows: {0}'.format(df_exp_cut.shape[0]))

    if complete_agg:
        agg_columns = [up, dn]
    else:
        agg_columns = [up, dn, 'pert_type', 'cell_id', 'pert_idose', 'pert_itime', 'is_touchstone']
    dfe = df_exp_cut.groupby(agg_columns).apply(lambda x:
                                                pd.Series([np.mean(x['score']), np.std(x['score'])],
                                                          index=['mean', 'std'])).reset_index()

    if verbose:
        print('df_exp_cut_agg rows: {0}'.format(dfe.shape[0]))

    dfe[cexp] = dfe['mean'].apply(lambda x: norm.cdf(x))
    if verbose:
        print('min score {0}, max score {1}'.format(df_exp_cut['score'].min(), df_exp_cut['score'].max()))
        print('min mean {0}, max mean {1}'.format(dfe['mean'].min(), dfe['mean'].max()))
        print('min cexp {0}, max cexp {1}'.format(dfe[cexp].min(), dfe[cexp].max()))
        print('size of experimental df {0}'.format(dfe.shape))
    dfe = dfe[o_columns + [cexp, 'std']]
    dfe2 = pd.merge(dfe, df_stats.reset_index(), on=o_columns, how='left')
    if verbose:
        print('experimental statements after agg and merge rows: {0}'.format(dfe2.shape[0]))

    merge_on = [ni]
    if not load_batches and hash_int and up in df_claims.columns and dn in df_claims.columns:
        merge_on = o_columns

    if (ni in dfe2.columns) and (ni in df_claims.columns):
        columns = list(set(dfe2.columns) - {ni})
    else:
        columns = dfe2.columns

    dft = df_claims.merge(dfe2[columns], on=merge_on, how='inner')

    if verbose:
        print('###')
        print('claims: before: {0} after: {1}  fraction: {2:.4f}'.format(df_claims.shape[0], dft.shape[0],
                                                                         dft.shape[0]/df_claims.shape[0]))
        n_ints = df_claims.drop_duplicates([up, dn]).shape[0]
        n_ints_after = dft.drop_duplicates([up, dn]).shape[0]
        print('interactions: before: {0} after: {1}  fraction: {2:.4f}'.format(n_ints, n_ints_after,
                                                                               n_ints_after/n_ints))

    # add static communities

    types_comm = ['lincs', 'litgw']
    # storage_types = ['csv.gz', 'h5']
    fpath_comm = expanduser('~/data/kl/comms/')
    up_dns = dft.drop_duplicates([up, dn])[[up, dn]]

    all_ids = set(dft[up].unique()) | set(dft[dn].unique())

    for ty in types_comm:
        fnames, cnames = get_community_fnames_cnames(ty)
        if verbose:
            print(len(fnames), fnames, cnames)
        for fn, cn in zip(fnames, cnames):
            dfc = pd.read_csv(join(fpath_comm, fn),
                              compression='gzip', index_col=0)
            comm_ids = set(dfc.index)
            if verbose:
                print('{0} {1}. |ids_merge| : {2}. |ids_comm| {3}. |ids_merge - ids_comm| {4}.'.
                      format(fn, cn, len(all_ids), len(comm_ids), len(all_ids - comm_ids)))
            vc = dfc.groupby('comm_id').apply(lambda x: x.shape[0])
            dfc2 = dfc.merge(pd.DataFrame(vc), left_on='comm_id', right_index=True).rename(columns={0: 'csize'})
            dfc2_up = dfc2.rename(columns=dict([(c, cn + '_' + c + '_up') for c in dfc2.columns]))
            dfc2_dn = dfc2.rename(columns=dict([(c, cn + '_' + c + '_dn') for c in dfc2.columns]))
            up_dns = up_dns.merge(dfc2_up, how='left', left_on='up', right_index=True)
            up_dns = up_dns.merge(dfc2_dn, how='left', left_on='dn', right_index=True)
            up_dns[cn + '_same_comm'] = (up_dns[cn + '_comm_id_up'] == up_dns[cn + '_comm_id_dn'])
            up_dns[cn + '_eff_comm_size'] = (up_dns[cn + '_csize_up'] * up_dns[cn + '_csize_dn']) ** 0.5
            del up_dns[cn + '_comm_id_up']
            del up_dns[cn + '_comm_id_dn']
            del up_dns[cn + '_csize_up']
            del up_dns[cn + '_csize_dn']

    dft = dft.merge(up_dns, on=[up, dn], how='left')

    # add dynamic communities
    ty = 'litgw'

    fnames, cnames = get_community_fnames_cnames(ty, 'h5')
    up_dns_ye = dft.drop_duplicates([up, dn, ye])[[up, dn, ye]].sort_values([up, dn, ye])
    up_dns_ye_acc = dft.drop_duplicates([up, dn, ye])[[up, dn, ye]].sort_values([up, dn, ye])

    for ff, cn in list(zip(fnames, cnames)):
        if verbose:
            print('filename of community {0}.'.format(ff))
        store = pd.HDFStore(expanduser(join(fpath_comm, ff)))
        keys = sorted(store.keys())
        dfa = []
        if verbose:
            print('keys of community h5 {0}.'.format(keys))
        for k in keys[:]:
            y = int(k[-4:])
            dfc = store.get(k)
            vc = dfc.groupby('comm_id').apply(lambda x: x.shape[0])
            dfc2 = dfc.merge(pd.DataFrame(vc), left_on='comm_id', right_index=True).rename(columns={0: 'csize'})
            dfc2_up = dfc2.rename(columns=dict([(c, cn + '_' + c + '_up') for c in dfc2.columns]))
            dfc2_dn = dfc2.rename(columns=dict([(c, cn + '_' + c + '_dn') for c in dfc2.columns]))
            ud_cur = up_dns_ye.loc[up_dns_ye[ye] == y].sort_values([up, dn, ye])
            ud_cur = ud_cur.merge(dfc2_up, how='left', left_on='up', right_index=True)
            ud_cur = ud_cur.merge(dfc2_dn, how='left', left_on='dn', right_index=True)
            ud_cur[cn + '_same_comm'] = (ud_cur[cn + '_comm_id_up'] == ud_cur[cn + '_comm_id_dn'])
            ud_cur[cn + '_eff_comm_size'] = (ud_cur[cn + '_csize_up'] * ud_cur[cn + '_csize_dn']) ** 0.5
            keep_cols = [c for c in ud_cur.columns if '_id_' not in c]
            dfa.append(ud_cur[keep_cols])
        up_dns_ye_acc = up_dns_ye_acc.merge(pd.concat(dfa), on=[up, dn, ye], how='left')
        store.close()

    dft = dft.merge(up_dns_ye_acc, on=[up, dn, ye], how='left')

    # add co-citation (future and past), coauthorship and co-affiliation metrics
    metric_sources = ['authors', 'affiliations', 'future', 'past']
    metric_types = ['support', 'affinity', 'linear', 'redmodularity']
    fpath_norm = '~/data/wos/cites/'
    fpath_alt = '~/data/wos/comm_metrics/'
    # metric_types = ['support', 'affinity', 'modularity']
    if verbose:
        print('support, affiliation, modularity metrics')
    # (*** here)
    for mt in metric_types:
        if mt == 'linear':
            metric_sources2 = metric_sources[:2]
        elif mt == 'redmodularity':
            metric_sources2 = metric_sources + ['afaupa', 'afaupafu']
        else:
            metric_sources2 = metric_sources
        if mt == 'redmodularity':
            fpath = fpath_alt
        else:
            fpath = fpath_norm
        for ms in metric_sources2:
            if mt == 'affinity' or mt == 'modularity' or mt == 'redmodularity':
                merge_cols = [up, dn, ye, pm]
            elif mt == 'support':
                merge_cols = [up, dn, ye]
            elif mt == 'linear':
                merge_cols = [up, dn, pm]
            else:
                raise ValueError('unsupported metric type')
            df_att = pd.read_csv(expanduser('{0}{1}_metric_{2}.csv.gz'.format(fpath, mt, ms)))
            if mt == 'modularity' or mt == 'redmodularity':
                cols = list(set(df_att.columns) - {up, dn, ye, pm})
                rename_dict = {c: '{0}_{1}'.format(ms, c) for c in cols}
            elif mt == 'linear':
                cols = list(set(df_att.columns) - {up, dn, ye, pm})
                rename_dict = {c: c for c in cols}
            else:
                rename_dict = {c: '{0}_{1}'.format(ms, c) for c in df_att.columns if 'ind' in c}
            support_cols = [c for c in rename_dict.keys()] + merge_cols
            if verbose:
                print('shape and columns: {0} {1}'.format(df_att.shape, rename_dict))
            dft = dft.merge(df_att[support_cols].rename(columns=rename_dict), on=merge_cols, how='left')

    if verbose:
        print('after merge to claims: {0}'.format(dft.shape[0]))

    if verbose:
        print('dft null values of cexp: {0}'.format(sum(dft[cexp].isnull())))

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


def simple_stratify(df, statify_column, seed=0, ratios=None, verbose=False):
    """
    based on multi-class column stratify_column
    return stratified df
    """
    if ratios == 'original':
        return df
    else:
        np.random.seed(seed)
        vc = df[statify_column].value_counts()
        masks = [(df[statify_column] == v) for v in vc.index]
        sizes = list(vc)
        if not isinstance(ratios, (list, tuple)):
            if ratios:
                leftover = 1.0 - ratios
                ratios = [leftover/(vc.shape[0]-1)]*(vc.shape[0]-1) + [ratios]
            else:
                ratios = [1.0/vc.shape[0]]*vc.shape[0]
        if len(ratios) == vc.shape[0]:
            tentative_sizes = np.array([n/alpha for n, alpha in zip(sizes, ratios)])
        else:
            print('ratios len does is not equal to the number of classes : '
                  'len ratios {0}, value counts {1}'.format(len(ratios), vc.shape[0]))
        if verbose:
            print('ratios in the training : {0}'.format(ratios))
        optimal_index = np.argmin(tentative_sizes)
        size0 = tentative_sizes[optimal_index]
        new_sizes = [int(x*size0) for x in ratios]
        # size = np.sum(new_sizes)
        indices = [np.random.choice(actual, new_size, replace=False) for actual, new_size in zip(sizes, new_sizes)]
        subdfs = [df.loc[m].iloc[ii] for m, ii in zip(masks, indices)]
        dfr = pd.concat(subdfs)
        return dfr


def simple_oversample(df, statify_column, rns, ratios=None, verbose=False):
    """
    return oversampled df
    """

    if not ratios:
        ratios = [1] * df[statify_column].unique()
    elif ratios == 'original':
        return df

    vc = df[statify_column].value_counts()
    if verbose:
        print(vc)
    masks = [(df[statify_column] == v) for v in vc.index]
    sizes = list(vc)

    new_sizes = [int(r * sizes[0] / ratios[0]) for r in ratios[1:]]
    if verbose:
        print('sizes: {0} new_size: {1}'.format(sizes, new_sizes))
    indices = [rns.choice(actual, new_size, replace=True) for
               actual, new_size in zip(sizes[1:], new_sizes)]
    # print([len(ii) for ii in indices])
    subdfs = [df.loc[masks[0]]] + [df.loc[m].iloc[ii] for m, ii in zip(masks[1:], indices)]
    dfr = pd.concat(subdfs)
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


def prepare_xy(df, covariate_columns, stratify=False, statify_size=5000,
               stratify_frac=0.5, seed=17, verbose=False,
               distance_column='guess'):

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
             seed=0, depth=None, fname=None, show=False, title_prefix=None, n_trees=13,
             return_model=False, class_weight='balanced_subsample', mode_scores=None):
    report = {}
    rf = RandomForestClassifier(n_trees, max_depth=depth,
                                random_state=seed,
                                class_weight=class_weight)
    n_states = len(set(y_test))

    rf = rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    report['corr_test_pred'] = np.corrcoef(y_pred, y_test)[0, 1]
    conf_matrix = confusion_matrix(y_test, y_pred)
    report['confusion'] = conf_matrix

    report['class_report'] = classification_report(y_test, y_pred)

    report['precision'] = precision_score(y_test, y_pred, average=mode_scores)
    report['accuracy'] = accuracy_score(y_test, y_pred)
    report['recall'] = recall_score(y_test, y_pred, average=mode_scores)
    report['f1'] = f1_score(y_test, y_pred, average=mode_scores)

    if n_states > 2:
        positive_proba = rf.predict_proba(X_test)
        y_test_binary = label_binarize(y_test, classes=np.arange(0.0, n_states))
        auroc = [roc_auc_score(y_, proba_) for proba_, y_ in zip(positive_proba.T, y_test_binary.T)]
    else:
        positive_proba = rf.predict_proba(X_test)[:, 1]
        auroc = roc_auc_score(y_test, positive_proba)
    report['auroc'] = auroc

    importances = rf.feature_importances_
    stds = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)

    report['feature_importance'] = dict(zip(covariate_columns, importances))
    report['feature_importance_std'] = dict(zip(covariate_columns, stds))

    fig, ax = plt.subplots(figsize=(5, 5))
    if n_states > 2:
        coords = [roc_curve(y_, proba_) for proba_, y_ in zip(positive_proba.T, y_test_binary.T)]
        for cs, auc, k in zip(coords, auroc, range(n_states)):
            fpr, tpr, thresholds = cs
            distance = k / (n_states - 1)
            ax.plot(fpr, tpr, label='Random Forest: dist {0} area = {1:.3f}'.format(distance, auc))
    else:
        fpr, tpr, thresholds = roc_curve(y_test, positive_proba)
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

    if return_model:
        return report, rf
    else:
        return report


def train_forest(dfw, feature_columns, y_column, seed, n_trees=10, mode_scores=None):
    return_model = True
    ratios = None
    show_plots = False
    title_prefix = None

    df_train, df_test = train_test_split(dfw, test_size=0.3,
                                         random_state=seed, stratify=dfw[y_column])
    X_test, y_test = df_test[feature_columns], df_test[y_column]
    X_train, y_train = df_train[feature_columns], df_train[y_column]

    r, rf_clf = rf_study(X_train, X_test, y_train, y_test, feature_columns, seed,
                         None, None, show_plots, title_prefix, n_trees, return_model,
                         mode_scores)
    return r, rf_clf


def train_massif(dfw, feature_columns, y_column,
                 seed, n_throws=10, n_trees=10, mode_scores=None, show_plots=False, fname=None,
                 ratios=None, title_prefix=None):
    return_model = True
    massif = []

    df_train, df_test = train_test_split(dfw, test_size=0.3,
                                         random_state=seed, stratify=dfw[y_column])
    X_test, y_test = df_test[feature_columns], df_test[y_column]

    n_states = len(set(y_test))
    seeds = sorted(np.random.choice(10000, n_throws, False))

    for seed in seeds:
        dd = simple_stratify(df_train, y_column, seed, ratios=ratios)
        X_train, y_train = dd[feature_columns], dd[y_column]

        r, rf_clf = rf_study(X_train, X_test, y_train, y_test, feature_columns, seed,
                             None, None, False, title_prefix, n_trees, return_model)

        massif.append(rf_clf)
    probs = [clean_zeros(rfm.predict_proba(X_test), 1e-2)[np.newaxis, ...] for rfm in massif]
    arr_probs = np.concatenate(probs, axis=0)
    rprobs = np.sum(arr_probs, axis=0)
    rprobs2 = rprobs / np.sum(rprobs, axis=1).reshape(-1)[:, np.newaxis]
    y_pred2 = np.argmax(rprobs2, axis=1)
    # rprobs3 = np.exp(np.sum(np.log(arr_probs), axis=0))
    # y_pred3 = np.argmax(rprobs3, axis=1)
    y_pred = y_pred2
    report = {}
    report['corr_test_pred'] = np.corrcoef(y_pred, y_test)[0, 1]
    conf_matrix = confusion_matrix(y_test, y_pred)
    report['confusion'] = conf_matrix

    report['class_report'] = classification_report(y_test, y_pred)

    report['precision'] = precision_score(y_test, y_pred, average=mode_scores)
    report['accuracy'] = accuracy_score(y_test, y_pred)
    report['recall'] = recall_score(y_test, y_pred, average=mode_scores)
    report['f1'] = f1_score(y_test, y_pred, average=mode_scores)

    if n_states > 2:
        positive_proba = rprobs2
        y_test_binary = label_binarize(y_test, classes=np.arange(0.0, n_states))
        auroc = [roc_auc_score(y_, proba_) for proba_, y_ in zip(positive_proba.T, y_test_binary.T)]
    else:
        positive_proba = rprobs[:, 1]
        auroc = roc_auc_score(y_test, positive_proba)

    report['auroc'] = auroc

    # importances = rf.feature_importances_
    # stds = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)

    importances = np.array([rf.feature_importances_ for rf in massif])
    stds = np.array([np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0) for rf in massif])

    report['feature_importance'] = {}
    report['feature_importance_std'] = {}

    for k, rvs, errors in zip(feature_columns, importances.T, stds.T):
        ls = np.ones(len(rvs))
        mean = np.mean(rvs)
        error = std_over_samples(ls, rvs, errors)
        report['feature_importance'][k] = mean
        report['feature_importance_std'][k] = error

    sns.set_style("whitegrid")
    if show_plots:
        fig, ax = plt.subplots(figsize=(5, 5))
        if n_states > 2:
            coords = [roc_curve(y_, proba_) for proba_, y_ in zip(positive_proba.T,
                                                                  y_test_binary.T)]
            for cs, auc, k in zip(coords, auroc, range(n_states)):
                fpr, tpr, thresholds = cs
                distance = k / (n_states - 1)
                ax.plot(fpr, tpr, label='Random Forest: dist {0} '
                                        'area = {1:.3f}'.format(distance, auc))
        else:
            fpr, tpr, thresholds = roc_curve(y_test, positive_proba)
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
    return report, massif, df_test


def train_massif_clean(df_train, feature_columns, y_column,
                       seed_=11, n_throws=10, n_trees=10, ratios=None,
                       min_samples_leaf=10):
    massif = []

    rns = RandomState(seed_)
    seeds = rns.randint(10000, size=n_throws)

    for seed_ in seeds:
        dd = simple_stratify(df_train, y_column, seed_, ratios=ratios)
        X_train, y_train = dd[feature_columns], dd[y_column]
        rf_clf = RandomForestClassifier(n_trees, max_depth=None, random_state=seed_, min_samples_leaf=min_samples_leaf)
        rf_clf = rf_clf.fit(X_train, y_train)
        massif.append(rf_clf)
    report = {}

    importances = np.array([rf.feature_importances_ for rf in massif])
    stds = np.array([np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0) for rf in massif])

    report['feature_importance'] = {}
    report['feature_importance_std'] = {}

    for k, rvs, errors in zip(feature_columns, importances.T, stds.T):
        ls = np.ones(len(rvs))
        mean = np.mean(rvs)
        error = std_over_samples(ls, rvs, errors)
        report['feature_importance'][k] = mean
        report['feature_importance_std'][k] = error

    return report, massif


def train_massif_lr_clean(df_train, feature_columns, y_column,
                           seed_=11, n_throws=10, n_trees=10, ratios=None):
    massif = []

    rns = RandomState(seed_)
    seeds = rns.randint(10000, size=n_throws)

    for seed_ in seeds:
        dd = simple_stratify(df_train, y_column, seed_, ratios=ratios)
        X_train, y_train = dd[feature_columns], dd[y_column]
        rf_clf = LogisticRegression(n_trees, random_state=seed_)
        rf_clf = rf_clf.fit(X_train, y_train)
        massif.append(rf_clf)
    report = {}

    importances = np.array([rf.feature_importances_ for rf in massif])
    stds = np.array([np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0) for rf in massif])

    report['feature_importance'] = {}
    report['feature_importance_std'] = {}

    for k, rvs, errors in zip(feature_columns, importances.T, stds.T):
        ls = np.ones(len(rvs))
        mean = np.mean(rvs)
        error = std_over_samples(ls, rvs, errors)
        report['feature_importance'][k] = mean
        report['feature_importance_std'][k] = error

    return report, massif


def plot_importances(importances, stds, covariate_columns, fname=None, title_prefix=None, colors=None,
                     show=False, ax=None, topn=20, sort_them=False):
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

    if sort_them:
        indices = np.argsort(np.abs(importances))[::-1]
    else:
        indices = list(range(importances.size))
    if topn:
        indices = indices[:topn]
    n = topn if topn else len(covariate_columns)
    if not colors:
        colors = 'r'
    imp_ccs = [covariate_columns[i] for i in indices]
    fig = plt.figure(figsize=(n*3, 5))
    sns.set_style("whitegrid")
    plt.title('{0} Random Forest feature importances'.format(title_prefix))
    importances2 = importances[indices]
    stds2 = stds[indices]
    sorted_ix = np.argsort(importances2)[::-1]
    colors = ['b' if x > 0 else 'r' for x in importances2[sorted_ix]]
    imp_ccs2 = [imp_ccs[i] for i in sorted_ix]

    plt.bar(range(len(importances2)), importances2[sorted_ix],
            color=colors, yerr=stds2[sorted_ix], align='center', alpha=0.5)
    # sns.barplot(list(range(n)), importances[indices],
    #             color=colors, yerr=stds[indices], align='center', alpha=0.5)

    plt.xticks(range(n), imp_ccs2)
    plt.xlim([-1, n])
    if fname:
        fig.savefig(fname)
        # if show:
        #     plt.show()
        # else:
        #     plt.close()


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
            fname=None, show=False, title_prefix=None):

    report = {'reg': regularizer}
    logreg = LogisticRegression(C=1. / regularizer, tol=1e-6, penalty='l1', fit_intercept=True,
                                random_state=seed, warm_start=True)
    logreg = logreg.fit(X_train, y_train)

    coef_report = dict(zip(covariate_columns, logreg.coef_[0]))
    coef_report['intercept'] = logreg.intercept_[0]
    report['coeffs'] = coef_report

    y_pred = logreg.predict(X_test)
    report['corr_train_pred'] = np.corrcoef(y_pred, y_test)[0, 1]

    report['precision'] = precision_score(y_test, y_pred)
    report['accuracy'] = accuracy_score(y_test, y_pred)
    report['recall'] = recall_score(y_test, y_pred)
    report['f1'] = f1_score(y_test, y_pred)

    conf_matrix = confusion_matrix(y_test, y_pred)
    report['confusion'] = conf_matrix
    report['class_report'] = classification_report(y_test, y_pred)

    positive_proba = logreg.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, positive_proba)
    report['auroc'] = roc_auc_score(y_test, positive_proba)

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


def lr_study_one_sample(X_train, X_test, y_train, y_test, covariate_columns=[], seed=0):

    reg_coeffs = 1e1**np.arange(-2, 2, 0.1)
    reports = []
    for coeff in reg_coeffs:
        report = lr_study(X_train, X_test, y_train, y_test, covariate_columns, seed, coeff)
        reports.append(report)

    optimal_report_index = pick_good_report(reports)

    columns_extra = covariate_columns + ['intercept']

    # array n_regularize x n_features
    # for a given j, arr[j, m_feature] is zero if m_feature is not important
    # the ranking of feature is due to them becoming non zero

    arr = np.array([[r['coeffs'][c] for c in columns_extra] for r in reports])

    # indices as which a given feature non negative, higher index,
    # higher penalty ~ more important feature
    nonzero_indices = np.argmax(arr == 0, axis=0)
    ranking = np.argsort(nonzero_indices)[::-1]

    optimal_report = reports[optimal_report_index]

    optimal_report['feature_ranking'] = ranking
    # optimal_report['']
    return optimal_report


def pick_good_report(reports):
    j_acc = np.argmax([r['accuracy'] for r in reports])
    j_rec = np.argmax([r['recall'] for r in reports])
    j_prec = np.argmax([r['precision'] for r in reports])
    j_f2 = np.argmax([r['f1'] for r in reports])
    # first occurence of threshold non zero coeffs
    j_optimal = min([j_acc, j_rec, j_prec, j_f2])
    return j_optimal


def run_n_lr_studies(df, feature_columns, y_column, n_throws=50, seed=0, ratios=(1., 2.)):
    np.random.seed(seed)
    seeds = sorted(np.random.choice(100, n_throws, False))

    agg_best_reports = []
    k = 0
    for seed in seeds:
        df_train, df_test = train_test_split(df, test_size=0.3,
                                             random_state=seed, stratify=df[y_column])
        dd = simple_stratify(df_train, y_column, seed, ratios=ratios)
        X_train, y_train = dd[feature_columns], dd[y_column]
        X_test, y_test = df_test[feature_columns], df_test[y_column]
        r = lr_study_one_sample(X_train, X_test, y_train, y_test, feature_columns, seed)
        r['seed'] = seed
        r['corr_train'] = dd[feature_columns + [y_column]].corr().iloc[-1].iloc[:-1].values
        r['corr_test'] = df_test[feature_columns + [y_column]].corr().iloc[-1].iloc[:-1].values
        r['vc_train'] = y_train.value_counts().to_dict()
        r['vc_test'] = y_test.value_counts().to_dict()
        r['size_train'] = y_train.shape[0]
        r['size_test'] = y_test.shape[0]
        r['test_frac'] = r['size_test'] / (r['size_test'] + r['size_train'])
        agg_best_reports.append(r)
        k += 1
    return agg_best_reports


def run_lr_over_list_features(df, features_list, distance_column, n_throws=20, seed=0):
    meta_report = []
    for feature_columns in features_list:
        ccs_extra = feature_columns + ['intercept']
        reports = run_n_lr_studies(df, feature_columns, distance_column, n_throws, seed)
        meta_report.append((ccs_extra, reduce_report(reports)))
    return meta_report


def reduce_report(report):
    final_report = {}
    r_sample = report[0]
    kkeys = r_sample.keys()
    for k in kkeys:
        if isinstance(r_sample[k], (int, float)):
            vector_c = [r[k] for r in report]
            final_report.update({k: (np.mean(vector_c), np.std(vector_c))})
        elif isinstance(r_sample[k], np.ndarray) and ('ranking' in k):
            final_report[k] = (np.array([r['feature_ranking'] for r in report]).mean(axis=0),
                               np.array([r['feature_ranking'] for r in report]).std(axis=0))
        else:
            final_report[k] = r_sample[k]
    return final_report


def parse_experiments_reports(reports):
    output_dict = {}
    index = []
    for k, dd in reports:
        name = '_'.join(k)
        index.append(name)
        for kk in dd.keys():
            # hunt down all tuples
            if isinstance(dd[kk], tuple) and (len(dd[kk]) == 2) and isinstance(dd[kk][0], float):
                if kk in output_dict.keys():
                    output_dict[kk].append(dd[kk][0])
                else:
                    output_dict[kk] = [dd[kk][0]]
                if kk+'_std' in output_dict.keys():
                    output_dict[kk+'_std'].append(dd[kk][1])
                else:
                    output_dict[kk+'_std'] = [dd[kk][1]]
    report_df = pd.DataFrame(output_dict, index=index)
    return report_df


def std_over_samples(lengths, means, stds):
    # https://stats.stackexchange.com/questions/43159/how-to-calculate-pooled-variance-of-two-groups-given-known-group-variances-mean
    # https://stats.stackexchange.com/questions/30495/how-to-combine-subsets-consisting-of-mean-variance-confidence-and-number-of-s
    total_length = np.sum(lengths)
    total_mean = np.sum(list(map(lambda x: x[0]*x[1], zip(lengths, means))))/total_length
    total_num = np.sum(list(map(lambda x: x[0]*(x[1]**2 + x[2]**2), zip(lengths, means, stds))))
    r = (total_num/total_length - total_mean**2)**0.5
    return r


def invert_bayes(df_input, clf, feature_columns, p_min=1e-2, verbose=False, debug=None):

    df = df_input.copy()
    if isinstance(debug, np.ndarray):
        arr_probs = debug
    else:
        if isinstance(clf, list):
            p_min = np.float(5e-1/clf[0].n_estimators)
            probs = [rfm.predict_proba(df[feature_columns])[np.newaxis, ...] for rfm in clf]
            arr_probs = np.concatenate(probs, axis=0)
            arr_probs2 = clean_zeros(arr_probs, p_min, 2)
            arr_probs2 = np.sum(arr_probs, axis=0)
            arr_probs2 = arr_probs2 / np.sum(arr_probs2, axis=1)[..., np.newaxis]
        else:
            arr_probs = clf.predict_proba(df[feature_columns])
            arr_probs2 = clean_zeros(arr_probs, p_min)
    if verbose:
        print('names of feature columns:', feature_columns)
        print('probs shape: ', arr_probs.shape)
    # arr_probs2 = clean_zeros(arr_probs, p_min)
    arr_ps = df[ps].values
    tensor_claims = np.stack([1 - arr_ps, arr_ps])
    tensor_probs = np.stack([arr_probs2, arr_probs2[:, ::-1]])
    if verbose:
        print(tensor_claims.T.shape, tensor_probs.T.shape)
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


def aggregate_over_claims_new(df, groupby_columns=(up, dn)):
    pes = ['pe_{0}'.format(j) for j in range(3)]
    p_agg = df.groupby(groupby_columns).apply(lambda x: np.sum(np.log(x[pes]), axis=0))
    qcexp_pred = p_agg[pes].apply(lambda x: np.float(np.argmax(x.values)), axis=1)
    qcexp_val = df.groupby(groupby_columns).apply(lambda x: x[qcexp].unique()[0])
    return qcexp_val, qcexp_pred


def aggregate_over_claims_comm(dfn, dft_comm, groupby_columns=(up, dn),
                               groupby_columns2=(up, dn, pm), groupby_columns3=(up, dn, 'rcommid')):
    pes = ['pe_{0}'.format(j) for j in range(3)]
    dfn2 = dfn.merge(dft_comm, how='inner', on=groupby_columns2)
    dfn3 = dfn2.groupby(groupby_columns3).apply(lambda x: np.mean(np.log(x[pes]), axis=0)).reset_index()
    p_agg = dfn3.groupby(groupby_columns).apply(lambda x: np.sum(x[pes], axis=0))
    qcexp_pred = p_agg[pes].apply(lambda x: np.float(np.argmax(x.values)), axis=1)
    qcexp_val = dfn2.groupby(groupby_columns).apply(lambda x: x[qcexp].unique()[0])
    return qcexp_val, qcexp_pred


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
        # vector of relative distances
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
            print('Warning {0} classes predict {1} unique members'.format(n_classes, len(np.unique(args))))
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


def clean_zeros(arr, p_min, sumaxis=1, verbose=False):
    arr2 = arr.copy()
    arr2[arr2 == 0.0] = p_min
    if verbose:
        print(np.sum(arr2, axis=sumaxis)[:5])
    arr2 = arr2/np.sum(arr2, axis=sumaxis)[...,  np.newaxis]
    if verbose:
        print(np.sum(arr2, axis=sumaxis)[:5])
    return arr2

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


def get_covs(corr_df, key):
    l1, l2 = 'level_0', 'level_1'
    corr = corr_df[(corr_df[l1] == key) | (corr_df[l2] == key)]
    mm = (corr['level_0'] < ct)
    corr['covariate'] = corr[l1]
    corr.loc[~mm, 'covariate'] = corr.loc[~mm, l2]
    return_corr = corr[['covariate', 0]].sort_values(0)
    return return_corr


def yield_candidates(corrs, pi_columns, target_column=dist, critical_size=3, mode='both', verbose=False):
    """

    :param corrs:
    :param pi_columns:
    :param target_column:
    :param critical_size:
    :param mode: 'anti', 'target' or 'both'
    :return:
    """

    candidates = list(set(corrs.columns) - set(pi_columns + [target_column]))
    # print(cors.loc[candidates, dist].abs())

    if mode == 'anti' or mode == 'both':
        # anticorrelates with pi_columns: smaller better
        anti_corr = corrs[candidates].apply(lambda x: np.sum((x[pi_columns] + 1.) ** 2))
        top_anti_corrs = np.argsort(anti_corr)
        candidates_anti_corr = anti_corr[top_anti_corrs].index
        # print(anti_corr[top_anti_corrs])

    if mode == 'target' or mode == 'both':
        # non zero correlation with target_column : higher better
        target_corr = corrs.loc[tuple(candidates), target_column].abs()
        top_target_corrs = np.argsort(target_corr)
        candidates_target_corr = target_corr[top_target_corrs].index

    if mode == 'both':
        # list a: |DABC| list b: |ABCD|
        # in list a (anti_corr) the elements are ordered in descending preference
        # in list b (target_corr) the elements are ordered in ascending preference
        # intersect them to obtain D, D, DB, DABC

        intersections = np.array([len(set(candidates_anti_corr[:k]) & set(candidates_target_corr[-k:]))
                                  for k in range(len(candidates_anti_corr))])
        critical_index = np.argmax(intersections >= critical_size)
        candidates_batch = list(set(candidates_anti_corr[:critical_index])
                                & set(candidates_target_corr[-critical_index:]))
        if verbose:
            print(target_corr[candidates_batch], anti_corr[candidates_batch])

    elif mode == 'target':
        candidates_batch = list(candidates_target_corr[-critical_size:])
    elif mode == 'anti':
        candidates_batch = list(candidates_anti_corr[:critical_size])
    return candidates_batch


def engine(df, all_cols, target_column, func, max_iterations=8, score_name='accuracy', critical_size=3, verbose=False):
    current_columns = list(all_cols)
    corrs = df[current_columns + [target_column]].corr()

    current_features = []
    max_iterations = min([max_iterations, len(all_cols)])
    best_reports = []
    if verbose:
        print('max iter: {0}'.format(max_iterations))

    for k in range(max_iterations):
        mode = 'anti' if k > 0 else 'target'
        candidates = yield_candidates(corrs, current_features, target_column, critical_size, mode)
        if verbose:
            print('iter: {0}, candidates {1}'.format(k, candidates))
        experiments = [current_features + [c] for c in candidates]
        rr = func(df, experiments, target_column)
        best_candidate_index = np.argmax([r[1][score_name][0] for r in rr])
        if verbose:
            print('iter: {0}, {1} {2}, chose {3}'.format(k, score_name,
                                                         rr[best_candidate_index][1][score_name][0],
                                                         candidates[best_candidate_index]))

        best_reports.append(rr[best_candidate_index])
        current_features.append(candidates[best_candidate_index])
    return best_reports


def logreg_driver(origin, version, batchsize, cutoff_len, a, b, hash_int, max_depth):

    feauture_cols = [ai, ar]

    windows = [1, 2]
    cur_metric_columns = ['cpop', 'cden', 'cpoprc', 'cdenrc']
    cur_metric_columns_exp = cur_metric_columns + [c+str(w) for w in windows for c in cur_metric_columns]
    cur_metric_columns_exp_normed = [c + '_normed' for c in cur_metric_columns_exp]
    data_columns = [ni, pm, ye] + feauture_cols + cur_metric_columns_exp + cur_metric_columns_exp_normed + [ps]
    print(data_columns)

    df = generate_samples(origin, version, a, b, batchsize, cutoff_len, data_columns=data_columns, hash_int=hash_int)

    cols_norm = [ai, ar, ct] + cur_metric_columns_exp + cur_metric_columns_exp_normed +\
                ['pre_' + affs, 'nhi_' + affs, 'pre_' + aus, 'nhi_' + aus, 'year_off', 'year_off2']
    cols_norm_by_int = []

    df2 = prepare_final_df(df, normalize=True, columns_normalize=cols_norm,
                           columns_normalize_by_interaction=cols_norm_by_int, cutoff=0.1, verbose=False)

    print('***')
    print('correlations of normed columnds with dist:')
    print(df2[cols_norm + [dist]].corr()[dist].sort_values())

    print('***')
    print('value counts of the target variable:')
    print(df2[dist].value_counts())

    # example run
    # rr = run_lr_over_list_features(df2, [['cden', 'cden_normed']], dist)

    output = engine(df2, cols_norm, dist, run_lr_over_list_features, max_iterations=max_depth,
                    score_name='f1', verbose=True)

    df_reports = parse_experiments_reports(output)
    df_reports2 = df_reports.reset_index()[['accuracy', 'precision',
                                            'recall', 'corr_train_pred',
                                            'f1', 'auroc', 'index']].sort_values('f1', ascending=False)

    fpath = expanduser('~/data/kl/reports/report_logreg_{0}_{1}_{2}.csv'.format(origin, version, hash_int))
    df_reports2.to_csv(fpath, float_format='%.3f')


def get_corrs(df, target_column, covariate_columns, threshold=0.03, mask=None,
              filename=None, filename_abs=None, dropnas=True, individual_na=False,
              verbose=False):

    df_ = df.copy()
    if mask is not None:
        df_ = df_[mask].copy()

    if verbose:
        print('number of rows {0}, which is a fraction {1:.3f} of the original'.format(df_.shape[0],
                                                                                       df_.shape[0]/df.shape[0]))

    if isinstance(covariate_columns, dict):
        covariate_columns_dict = covariate_columns
        covariate_columns = list([x for sublist in covariate_columns.values() for x in sublist])
        dict_flag = True
    else:
        covariate_columns_dict = dict()
        dict_flag = False

    all_cols = list(set(covariate_columns) | {target_column})
    if individual_na:
        corrs = []
        for c in covariate_columns:
            mask = df_[c].notnull()
            if c in df.columns and sum(mask) > 2:
                corr_ = df_.loc[mask, [c, target_column]].corr()
                if corr_.shape == (2, 2):
                    corrs.append(corr_.values[0, 1])
                else:
                    corrs.append(np.nan)
            else:
                corrs.append(np.nan)
        corr_df = pd.Series(corrs, index=covariate_columns, name=target_column)
        corr_abs_thr = corr_df[corr_df.abs() > threshold].abs().sort_values(ascending=False)
        corr_df_thr = corr_df[corr_df.abs() > threshold].sort_values(ascending=False)
    else:
        corr_df = df_[all_cols].corr()
        not_na_mask = corr_df[target_column].notnull()
        not_na_columns = list(not_na_mask[not_na_mask].index)
        if not dropnas:
            not_na_columns = list(corr_df[target_column].index)
        corr_df_abs = corr_df.abs()

        reduced_columns = []
        if dict_flag:
            for k, v in covariate_columns_dict.items():
                cur_cols = list(set(not_na_columns) & set(v))
                if len(cur_cols) > 1:
                    candidates = corr_df.loc[cur_cols, target_column].sort_values(ascending=False).iloc[[0, -1]]
                else:
                    candidates = corr_df.loc[cur_cols, target_column]
                # print(list(candidates.index))
                reduced_columns.extend(list(candidates.index))
        else:
            reduced_columns = list(set(covariate_columns) & set(not_na_columns))

        # print(reduced_columns)
        above_thr = set(corr_df_abs[corr_df_abs[target_column] > threshold].index)

        corr_abs_thr = corr_df_abs.loc[list(set(reduced_columns) & above_thr),
                                       target_column].sort_values(ascending=False)
        corr_df_thr = corr_df.loc[list(set(reduced_columns) & above_thr),
                                  target_column].sort_values(ascending=False)

    if filename:
        corr_df_thr.to_csv(filename)
    if filename_abs:
        corr_abs_thr.to_csv(filename_abs)

    if verbose:
        # print(corr_abs_thr)
        print(corr_df_thr)
    return corr_abs_thr, corr_df_thr


def trim_corrs_by_family(cr, feature_dict_inv):
    cr_name = cr.name
    if cr.shape[0] > 0:
        cr_df = cr.reset_index()
        cr_df['family'] = cr_df['index'].apply(lambda x: feature_dict_inv[x])
        mask_pos = (cr_df[cr_name] > 0)
        cr_df_cut_pos = cr_df.loc[mask_pos].groupby('family').apply(lambda x:
                                                                    x.iloc[np.argmax(x[cr_name].values)].append(
                                                                        pd.Series([x.shape[0]], ['len_members'])))
        cr_df_cut_neg = cr_df.loc[~mask_pos].groupby('family').apply(lambda x:
                                                                     x.iloc[np.argmin(x[cr_name].values)].append(
                                                                         pd.Series([x.shape[0]], ['len_members'])))
        if cr_df_cut_neg.shape[0] > 0 or cr_df_cut_pos.shape[0] > 0:
            cr_res = pd.concat([cr_df_cut_pos, cr_df_cut_neg]).set_index('index').sort_values(cr_name)
            unique_family_size = cr_res.drop_duplicates('family').shape[0]
            if unique_family_size != cr_res.shape[0]:
                confused_families = cr_res.groupby('family').apply(lambda x: x.shape[0] > 1)
                confused_families_mask = cr_res['family'].isin(confused_families[confused_families].index)
                print('Warning, a family of features has correlation of different signs!')
                print('size of extreme corrs dataset: {0}, size of unique families dataset {1}'.format(unique_family_size,
                                                                                                       cr_res.shape[0]))
                print('Printing confused families:')
                print(cr_res[confused_families_mask])
            return cr_res[['family', cr_name, 'len_members']]
        else:
            return pd.DataFrame(columns=['family', cr_name, 'len_members'])
    else:
        return pd.DataFrame(columns=['family', cr_name, 'len_members'])


def select_features_dict(df_train, df_test, target_column, feature_dict,
                         model_type='rf',
                         max_features_consider=8,
                         metric_mode='accuracy',
                         mode_scores=None,
                         metric_uniform_exponent=0.5,
                         eps_improvement=1e-6,
                         model_dict={},
                         max_feat_per_family=1,
                         verbose=False):
    """

    :param df_train: DataFrame for training
    :param df_test: DataFrame for testing
    :param target_column: target column
    :param feature_dict: dict of feature groups
    :param model_type: scikit-learn-like model type; rf or lr
    :param max_features_consider: maximum number of features of the final model
    :param metric_mode: which metric to choose for optimizing the model
                        from ['corr',  'accuracy', 'precision','recall', 'f1']
    :param mode_scores:
    :param metric_uniform_exponent: the exponent to make small metrics more pronounced (0.5 by default)
    :param eps_improvement: epsilon at which to stop improvment
    :param model_dict:
    :param verbose:
    :return:
    """

    y_train, y_test = df_train[target_column], df_test[target_column]

    chosen_features = []
    chosen_metrics = []
    chosen_total_metrics = []
    feature_dict_dyn = deepcopy(feature_dict)
    feature_dict_inv = {}
    for k, v in feature_dict.items():
        feature_dict_inv.update({x: k for x in v})

    if verbose:
        print('model_dict {0}'.format(model_dict))
    while len(chosen_features) <= max_features_consider and feature_dict_dyn:
        trial_features = [x for sublist in feature_dict_dyn.values() for x in sublist]
        scalar_metrics = []
        cur_metrics = []
        for tf in trial_features:
            cur_features = chosen_features + [tf]
            X_train, X_test = df_train[cur_features], df_test[cur_features]

            if model_type == 'rf':
                model = RandomForestClassifier(**model_dict)
            elif model_type == 'rfr':
                model = RandomForestRegressor(**model_dict)
            elif model_type == 'lr':
                model = LogisticRegression(**model_dict)
            elif model_type == 'lrg':
                model = LinearRegression(**model_dict)
            else:
                raise ValueError('model_type value is not admissible')

            model.fit(X_train, y_train)
            rmetrics = report_metrics(model, X_test, y_test, mode_scores, metric_uniform_exponent, metric_mode,
                                      problem_type=problem_type_dict[model_type])
            scalar_metrics.append(rmetrics['main_metric'])
            cur_metrics.append(rmetrics)

        add_index = np.nanargmax(np.array(scalar_metrics))
        if len(chosen_metrics) > 0:
            potential_improvement = np.sign(chosen_metrics[-1])*(1 - chosen_metrics[-1]/scalar_metrics[add_index])
            if verbose:
                print('Fractional potential improvement: {0:.3f}, abs value: {1:.3f}'.format(potential_improvement,
                                                                                             scalar_metrics[add_index]))
            if potential_improvement < eps_improvement:
                if verbose:
                    print('Terminating early: no improvement.')
                break
        else:
            potential_improvement = 1

        current_feature = trial_features[add_index]
        feature_group = feature_dict_inv[current_feature]

        chosen_metrics.append(scalar_metrics[add_index])

        chosen_features.append(current_feature)
        chosen_total_metrics.append(cur_metrics[add_index])

        if verbose:
            print('nf: {0} cfeature: {1} metric: {2:.3f} metric_improv: {3:.2f} %'.format(len(cur_features),
                  (current_feature[:47]+'...').ljust(50),
                  chosen_metrics[-1], 100*potential_improvement))

        if len(feature_dict_dyn[feature_group]) <= len(feature_dict[feature_group]) - (max_feat_per_family - 1):
            if verbose:
                str_rep = 'in dyn dict {0} in tot dict {1}'.format(len(feature_dict_dyn[feature_group]),
                                                                   len(feature_dict[feature_group]))
                print('Feature group {0}: {1}'.format(feature_group, str_rep))
            del feature_dict_dyn[feature_group]
        else:
            feature_dict_dyn[feature_group].remove(current_feature)

    if model_type == 'rf':
        model = RandomForestClassifier(**model_dict)
    elif model_type == 'rfr':
        model = RandomForestRegressor(**model_dict)
    elif model_type == 'lr':
        model = LogisticRegression(**model_dict)
    elif model_type == 'lrg':
        model = LinearRegression(**model_dict)
    else:
        raise ValueError('model_type value is not admissible')

    X_train = df_train[chosen_features]
    model.fit(X_train, y_train)
    return chosen_features, chosen_metrics, chosen_total_metrics, model


def report_metrics(model, X_test, y_test, mode_scores=None, metric_uniform_exponent=0.5, metric_mode='accuracy',
                   problem_type='class'):

    y_pred = model.predict(X_test)

    report = report_metrics_(y_test, y_pred, mode_scores, metric_uniform_exponent, metric_mode,
                             problem_type)
    if problem_type == 'class':
        nclasses = len(set(y_test))
        positive_proba = model.predict_proba(X_test)
        y_test_binary = label_binarize(y_test, classes=np.arange(0.0, nclasses))
        auroc = [roc_auc_score(y_, proba_) for proba_, y_ in zip(positive_proba.T, y_test_binary.T)]
        report['auroc'] = auroc
    return report


def report_metrics_(y_test, y_pred, mode_scores=None, metric_uniform_exponent=0.5, metric_mode='accuracy',
                    problem_type='class'):

    report = dict()
    # NB corr might be NaN
    report['corr'] = np.corrcoef(y_pred, y_test)[0, 1]
    if problem_type == 'class':
        report['accuracy'] = dict()
        report['accuracy']['normal'] = accuracy_score(y_test, y_pred)
        report['accuracy']['balanced'] = balanced_accuracy_score(y_test, y_pred, adjusted=True)

        report['vector'] = dict()
        report['vector']['precision'] = precision_score(y_test, y_pred, average=mode_scores)
        report['vector']['recall'] = recall_score(y_test, y_pred, average=mode_scores)
        report['vector']['f1'] = f1_score(y_test, y_pred, average=mode_scores)

        report['macro'] = dict()
        report['macro']['precision'] = precision_score(y_test, y_pred, average='macro')
        report['macro']['recall'] = recall_score(y_test, y_pred, average='macro')
        report['macro']['f1'] = f1_score(y_test, y_pred, average='macro')

        nclasses = len(set(y_test))

        report['exponent'] = dict()
        report['exponent'] = {k: np.sum(v ** metric_uniform_exponent)/nclasses for k, v in report['vector'].items()}

        if metric_mode == 'accuracy':
            report['main_metric'] = report['accuracy']['balanced']
        else:
            report['main_metric'] = report['macro'][metric_mode]

        report['conf'] = confusion_matrix(y_test, y_pred)

    else:
        report['mse'] = np.sum((1 - y_test/y_pred)**2)**0.5/y_pred.shape[0]
        report['main_metric'] = -report['mse']
    return report


def logit_pvalue(model, x, verbose=False):
    """ Calculate z-scores for scikit-learn LogisticRegression.
    parameters:
        model: fitted sklearn.linear_model.LogisticRegression with intercept and large C
        x:     matrix on which the model was fit
    """
    probs = model.predict_proba(x)
    n_datapoints = probs.shape[0]
    n_feautures = len(model.coef_[0]) + 1
    coeffs = np.hstack([model.intercept_.reshape(-1, 1), model.coef_])
    x_full = np.matrix(np.insert(np.array(x), 0, 1, axis=1))
    pvals = []
    errors = []
    for coeffs_vec, p_vec in zip(coeffs, probs.T):
        ans = np.zeros((n_feautures, n_feautures))
        for i in range(n_datapoints):
            ans += np.dot(x_full[i].T, x_full[i]) * p_vec[i]*(1 - p_vec[i])
        try:
            vcov = np.linalg.inv(np.matrix(ans))
            serrors = np.sqrt(np.diag(vcov))
            t = coeffs_vec / serrors
            pn = (1 - norm.cdf(abs(t))) * 2
        except np.linalg.linalg.LinAlgError as e:
            if verbose:
                print('det : {0}'.format(np.linalg.det(np.matrix(ans))))
            serrors = np.zeros(ans.shape[0])
            pn = np.zeros(ans.shape[0])
        pvals.append(pn)
        errors.append(serrors)
    pvals = np.array(pvals)
    errors = np.array(errors)
    return pvals.T, coeffs.T, errors.T


def linear_pvalue(model, X, y):
    """ Calculate z-scores for scikit-learn LinearRegression.
    parameters:
        model: fitted sklearn.linear_model.LinearRegression with intercept and large C
        x:     matrix on which the model was fit
    """

    pred = model.predict(X)
    sse = np.sum((pred - y) ** 2, axis=0)
    full_X = np.hstack([np.ones((X.shape[0], 1)), X])
    sse = sse / (float(full_X.shape[0] - full_X.shape[1]))
    full_co = np.array([model.intercept_] + list(model.coef_))
    try:
        vcov = np.linalg.inv(np.dot(full_X.T, full_X))
        se = np.sqrt(np.diagonal(sse * vcov))
        t = full_co / se
        pvals = 2 * (1 - tdistr.cdf(np.abs(t), y.shape[0] - full_X.shape[1]))
    except np.linalg.linalg.LinAlgError as e:
        se = np.zeros(X.shape[1])
        pvals = np.zeros(X.shape[1])
    return pvals.T, full_co.T, se.T
