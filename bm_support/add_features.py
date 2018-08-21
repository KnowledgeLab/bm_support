import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler
from datahelpers.dftools import agg_file_info
from os import listdir
from os.path import isfile, join, expanduser
from datahelpers.constants import iden, ye, ai, ps, up, dn, ar, ni, cexp, qcexp, dist, rdist, pm, ct, affs, aus
from datahelpers.dftools import select_appropriate_datapoints, add_column_from_file
import Levenshtein as lev
from functools import partial
from .disambiguation import ndistance, nlevenstein_root, cluster_objects
from sklearn.model_selection import train_test_split
from numpy import unique
import gzip
import pickle


def gcd(a, b):
    if b == 0:
        return a
    remainder = a % b
    return gcd(b, remainder)


def lcm(a, b):
    return a * b / gcd(a, b)


def groupby_normalize(data):
    data_ = data.values
    std = data_.std()
    if np.abs(std) < 1e-12:
        data_rel = data_ - data_.mean()
    else:
        data_rel = (data_ - data_.mean())/std
    if np.isnan(std):
        print(data_.shape[0], data.index, std)

    return pd.Series(data_rel, index=data.index)


def normalize_columns(df, columns):
    df2 = df.copy()
    sc = MinMaxScaler()
    df2[columns] = sc.fit_transform(df[columns])
    return df2


def mask_out(df, cutoff=None, extra_masks=None, verbose=False):
    masks = []

    if extra_masks:
        masks.extend(extra_masks)

    # mask only only the upper and the lower quartiles in cdf_exp
    if cutoff:
        upper_exp, lower_exp = 1 - cutoff, cutoff
        exp_mask = ['cdf_exp', (upper_exp, lower_exp), lambda df_, th: (df_ >= th[0]) | (df_ <= th[1])]
        masks.append(exp_mask)

    # mask affiliation rating
    # ar < 0 means affiliation rating was not present ???
    # ar_mask = [ar, 0., lambda df_, th: (df_ >= th)]
    # masks.append(ar_mask)

    # mask article influence
    # ai equal to the top of value_counts() means that it was imputed
    # ai_mask = [ai, 0., lambda s, th: (s != s.value_counts().index[0])]
    # masks.append(ai_mask)

    # duplicate claims from the same journal are encoded as -1
    # mask them out
    ps_mask = [ps, 0., lambda s, th: (s >= th)]
    masks.append(ps_mask)

    df_selected = select_appropriate_datapoints(df, masks)

    if verbose:
        print('received: {0} rows, after masking out: {1} rows.'.format(df.shape[0], df_selected.shape[0]))
    return df_selected


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


def define_distance_(df, columns, verbose=False):
    """
    df contains columns a and b
    with values 0, 1, .., k and 0, 1, .., m respectively
    both are expanded to  0, 1 ... lcm(k, m)
    the distance is
    """
    if len(columns) != 2:
        raise ValueError('in define_distance() columns argument is not length two')
    elif not (set(columns) < set(df.columns)):
        raise ValueError('in define_distance() columns are not in df.columns')

    a, b = columns
    n_a = df[a].value_counts().shape[0] - 1
    n_b = df[b].value_counts().shape[0] - 1
    lcm_ab = lcm(n_a, n_b)
    m_a = lcm_ab / n_a
    m_b = lcm_ab / n_b
    if verbose:
        print('class a scale: {0}; class b scale: {1}. lcm {2}'.format(n_a, n_b, lcm_ab))
        print('m a {0}; m b {1}'.format(m_a, m_b))

    s = np.abs(m_b * df[b] - m_a * df[a])
    if verbose:
        print(s.value_counts(), s.mean())
    return s


def derive_distance_column(df, column_a_parameters=(cexp, qcexp, (-1.e-8, 0.5, 1.0)),
                           column_b_parameters=ps,
                           distance_column='guess', verbose=False):
    cols = []
    for par in (column_a_parameters, column_b_parameters):
        if isinstance(par, tuple) and not isinstance(par, str):
            c, qc, thrs = par
            df[qc] = quantize_series(df[c], thrs, verbose)
            cols.append(qc)
        else:
            cols.append(par)

    df[distance_column] = define_distance_(df, cols, verbose)
    return df


def prepare_final_df(df, normalize=False, columns_normalize=None, columns_normalize_by_interaction=None,
                     cutoff=None, quantize_intervals=(-1.e-8, 0.5, 1.0), aff_dict_fname=None,
                     suppress_affs=False,
                     masks=None,
                     add_cite_fits=False,
                     min_len=-1,
                     max_len=None,
                     verbose=False):

    mask_len_ = (df.groupby(ni).apply(lambda x: x.shape[0]) > min_len)
    mask_len = df[ni].isin(mask_len_[mask_len_].index)
    df = df[mask_len]
    if max_len:
        mask_len_ = (df.groupby(ni).apply(lambda x: x.shape[0]) < max_len)
        mask_len = df[ni].isin(mask_len_[mask_len_].index)
        df = df[mask_len]

    df_selected = mask_out(df, cutoff, masks, verbose)

    pmids = df_selected[pm].unique()

    df_wos = retrieve_wos_aff_au_df()

    if verbose:
        print('defining authors\' features')
    # {pmid : [i_A]} -  dict of lists of of authors
    pm_aus_map = cluster_authors(df_wos, pmids)
    # clean {pmid : [i_A]} of possible duplicates
    pm_aus_map = {k: list(set(v)) for k, v in pm_aus_map.items()}

    # for each new_index ni (interaction) calculate authors herfindahl index
    aus_feature = df_selected.groupby(ni).apply(lambda x: calc_normed_hi(x, pm_aus_map, (pm, ye)))
    aus_feature = aus_feature.rename(columns={0: pm, 1: 'pre_' + aus, 2: 'nhi_' + aus})
    f_cols = ['pre_' + aus, 'nhi_' + aus]

    if not suppress_affs:
        if verbose:
            print('defining affiliations\' features...')
        if aff_dict_fname:
            with gzip.open(aff_dict_fname, 'rb') as fp:
                pm_clusters = pickle.load(fp)
            if verbose:
                print('loaded pm_clusters, dict contains {0} pmids'.format(len(pm_clusters)))
        else:
            pm_clusters = cluster_affiliations(df_wos, pmids)

        pm_clusters = {k: list(set(v)) for k, v in pm_clusters.items()}
        affs_feature = df_selected.groupby(ni).apply(lambda x: calc_normed_hi(x, pm_clusters, (pm, ye)))
        affs_feature = affs_feature.rename(columns={0: pm, 1: 'pre_' + affs, 2: 'nhi_' + affs})
        aus_feature = affs_feature.merge(aus_feature[['pre_' + aus, 'nhi_' + aus]],
                                         left_index=True, right_index=True)
        f_cols.extend(['pre_' + affs, 'nhi_' + affs])

    df_selected = df_selected.merge(aus_feature[f_cols],
                                    left_index=True, right_index=True)

    df_selected[ye + '_off'] = df_selected.groupby(ni, as_index=False,
                                                   group_keys=False).apply(lambda x: x[ye] - x[ye].min())
    df_selected[ye + '_off2'] = df_selected[ye + '_off']**2

    # add citation count
    fp = expanduser('~/data/literome/wos/pmid_wos_cite.csv.gz')
    dft2_ = add_column_from_file(df_selected, fp, pm, ct)

    # define distance between qcexp and ps
    dft2 = derive_distance_column(dft2_, (cexp, qcexp, quantize_intervals), ps, dist)
    dft2[rdist] = (dft2[cexp] - dft2[ps]).abs()

    dft = dft2
    if verbose:
        print('value counts of distance:')
        print(dft2[dist].value_counts())

    if add_cite_fits:
        df_cites = pd.read_csv(expanduser('~/data/wos/cites/wos_cite_result.csv.gz'),
                               compression='gzip', index_col=0)
        dft3 = pd.merge(dft2, df_cites, on=pm, how='left')
        for c in ['yearspan_flag', 'len_flag', 'succfit_flag', 'mu', 'sigma', 'A', 'err', 'int_3']:
            nan_mask = dft3[c].isnull()
            report_nas = sum(nan_mask)
            print('{0} of {1} entries were not identified in wos db ({2:.1f}%)'.format(report_nas, c,
                                                                                       100*report_nas/dft3.shape[0]))
            dft3.loc[nan_mask, c] = -1.0
        report_nas = sum(dft3['sigma'].isnull())

        mask = (dft3['A'] > 0)
        dft3['int_3_log'] = -1.0
        dft3['A_log'] = -1.0
        dft3['int_3_log_sigma'] = -1.0
        dft3['A_log_sigma'] = -1.0

        dft3.loc[mask, 'int_3_log'] = np.log(dft3.loc[mask, 'int_3'] + 1)
        dft3.loc[mask, 'A_log'] = np.log(dft3.loc[mask, 'A'] + 1)
        dft3.loc[mask, 'int_3_log_sigma'] = (dft3.loc[mask, 'int_3_log'] - dft3.loc[mask, 'int_3_log'].mean()) ** 2
        dft3.loc[mask, 'A_log_sigma'] = (dft3.loc[mask, 'A_log'] - dft3.loc[mask, 'A_log'].mean()) ** 2
        dft = dft3
        if verbose:
            print('{0} entries were not identified in wos db  ({1:.1f}%)'.format(report_nas,
                                                                                 100*report_nas/dft3.shape[0]))

    if normalize:
        if verbose:
            minmax = ['{0} min: {1:.2f}; max {2:.2f}'.format(c, dft[c].min(), dft[c].max())
                      for c in columns_normalize]
            print('. '.join(minmax))
        if columns_normalize:
            dft = normalize_columns(dft, columns_normalize)

        for c in columns_normalize_by_interaction:
            dft[c] = dft.groupby(ni, as_index=False, group_keys=False).apply(lambda x: groupby_normalize(x[c]))

        if verbose:
            minmax = ['{0} min: {1:.2f}; max {2:.2f}'.format(c, dft[c].min(), dft[c].max())
                      for c in columns_normalize]
            print('. '.join(minmax))

    return dft


def retrieve_wos_aff_au_df(fpath='~/data/wos/wos_pmid/', verbose=False):
    fpath = expanduser(fpath)
    suffix = 'txt'
    prefix = 'sav'
    suffix_len = len(suffix)
    prefix_len = len(prefix)
    files = [f for f in listdir(fpath) if isfile(join(fpath, f)) and
             (f[-suffix_len:] == suffix) and (f[:prefix_len] == prefix)]

    if verbose:
        print(files)

    kk = ['PM', 'TC', 'UT', 'AU', 'C1']
    ll = [agg_file_info(join(fpath, f), kk) for f in files]
    lll = [x for sublist in ll for x in sublist]

    df = pd.DataFrame(lll, columns=kk)
    df = df.rename(columns={'PM': pm, 'TC': ct, 'UT': 'wos_id',
                            'AU': aus, 'C1': affs})
    df[pm] = df[pm].astype(int)
    return df


def process_affs(df, verbose=False):

    # number of [] () inserts in affiliations
    if verbose:
        print('number of affs with authors in square brackets [] :'
              ' {0}'.format(sum(df[affs].apply(lambda x: '[' in x))))
    # we need to exclude [abc] and (abc), they contain authors in affiliation
    # df[affs] = df[affs].apply(lambda x: re.sub('[\(\[].*?[\)\]]', '', x))
    df[affs + '_clean'] = df[affs].apply(lambda x: re.sub('[\(\[].*?[\)\]]', '', x))
    # number of empty authors, empty affiliations
    if verbose:
        print('number of empty affs: {0}'.format(sum(df[affs].apply(lambda x: x == ''))))
    df_working = df.loc[df[affs + '_clean'].apply(lambda x: x != '')]

    pm_aff_map = df_working[[pm, affs + '_clean']].values

    pm_aff_split = [(pm_, x.split('|')) for pm_, x in pm_aff_map]

    pm_aff_phrases = []
    for pmid, phrase in pm_aff_split:
        phrase2 = [x.split(',')[0].replace('& ', '').strip().lower() for x in phrase]
        phrase2 = list(set(phrase2))
        pm_aff_phrases.append((pmid, phrase2))

    affs_list_lists = [x[1] for x in pm_aff_phrases]
    affs_lists = [x for sublist in affs_list_lists for x in sublist if x != '']
    affs_uniq = list(set(affs_lists))

    index_aff2 = list(zip(range(len(affs_uniq)), affs_uniq))
    index_aff3 = [(j, x.split()) for j, x in index_aff2]
    a2i = dict([(aff, j) for j, aff in index_aff2])

    return index_aff3, pm_aff_phrases, a2i


def cluster_affiliations(df, pmids=[], n_processes=1, verbose=False, debug=False):

    if len(pmids) > 0:
        df = df.loc[df[pm].isin(pmids)]

    index_affs, pm_aff_phrases, a2i = process_affs(df, verbose)

    ndist = partial(ndistance, **{'foo': lev.distance})
    ndist_root = partial(nlevenstein_root, **{'foo': ndist})

    i2c, phrase_cluster_dict = cluster_objects(index_affs[:],
                                               foo=ndist_root, foo_basic=ndist,
                                               n_processes=n_processes, simple_thr=0.1,
                                               verbose=verbose, debug=True)

    pm_clusters = {}
    for pmid, phrase in pm_aff_phrases:
        pm_clusters[pmid] = [i2c[a2i[p]] for p in phrase]

    if debug:
        return pm_clusters, phrase_cluster_dict
    else:
        return pm_clusters


def cluster_authors(df, pmids=[], verbose=False):
    """

    :param df:
    :param pmids:
    :param verbose:
    :return: dict {pmid: [i_A]}, where [i_A] is a list of author ids
    """
    if len(pmids) > 0:
        df = df.loc[df[pm].isin(pmids)]

    # number of [] () inserts in affiliations
    if verbose:
        print(sum(df[affs].apply(lambda x: '[' in x)))

    if verbose:
        print('number of empty authors: {0}',
              format(sum(df[aus].apply(lambda x: x == ''))))

    df_working = df.loc[df[aus].apply(lambda x: x != '')]
    pm_aus_map = df_working[[pm, aus]].values
    pm_aus_split = [(pm_, x.split('|')) for pm_, x in pm_aus_map]

    au_llist = [x[1] for x in pm_aus_split]
    au_list = [x for sublist in au_llist for x in sublist]
    aus_unique = list(set(au_list))
    if verbose:
        print('authors :', len(au_list), '. uniques:', len(set(au_list)))

    a2i = dict(zip(aus_unique, range(len(aus_unique))))
    pm_clusters = {}
    for pmid, phrase in pm_aus_split:
        pm_clusters[pmid] = [a2i[p] for p in phrase]

    return pm_clusters


def update_dict_numerically(d1, d2, normalization=None):
    # useful for herfindahl index
    if normalization:
        n1, n2 = normalization
        n = n1 + n2
    else:
        n1, n2, n = 1, 1, 1
    ret_dict = {k: n1*v/n for k, v in d1.items()}

    for k, v in d2.items():
        if k in ret_dict:
            ret_dict[k] += n2*v/n
        else:
            ret_dict[k] = n2*v/n
    return ret_dict


# calculation of normalized herfindahl index
def calc_normed_hi(pd_incoming, pmids_clust_dict, columns, verbose=False):
    # use agg_column to bin (discretize) index calculation
    id_column, agg_column = columns
    years = pd_incoming[agg_column]
    uniques, cnts = unique(years, return_counts=True)
    dict_pmcid = {}
    dict_year_pm = {}

    if verbose:
        print('u, c:', uniques, cnts)

    years2 = []
    for pmid, year in pd_incoming[[id_column, agg_column]].values:
        # for pmid present in clust dict
        if pmid in pmids_clust_dict.keys():
            # create {year: [pmid]} dict
            if year in dict_year_pm.keys():
                dict_year_pm[year].append(pmid)
            else:
                dict_year_pm[year] = [pmid]
            # array on units in pmid
            arr_pmids = pmids_clust_dict[pmid]
            # number of units per pmid
            length = len(arr_pmids)
            # each unit contributes as 1/number
            d2 = dict(zip(arr_pmids, [1. / length] * length))
            # accumulate {year: {id_k: frac_k}} dict
            if year in dict_pmcid.keys():
                dict_pmcid[year] = update_dict_numerically(dict_pmcid[year], d2)
            else:
                dict_pmcid[year] = d2
            years2.append(year)

    # update the stats of pmid occurences (wrt to present in clust_dict)
    uniques2, cnts2 = unique(years2, return_counts=True)

    if verbose:
        print(uniques2, cnts2)

    # normalize index by the number of items in each bin
    for year, cnt in zip(uniques2, cnts2):
        dict_pmcid[year] = {k: v / cnt for k, v in dict_pmcid[year].items()}
    cumsums2 = np.cumsum(cnts2)

    # accumulate the index; sum over preceding years
    dict_pmcid_acc = {}
    personal_frac = {}
    if uniques2.size > 0:
        dict_pmcid_acc[uniques2[0]] = dict_pmcid[uniques2[0]]
        for k, q, n1, n2 in zip(uniques2[:-1], uniques2[1:], cumsums2[:-1], cnts2[1:]):
            dict_pmcid_acc[q] = update_dict_numerically(dict_pmcid_acc[k], dict_pmcid[q], (n1, n2))

        hi = [(year, len(dict_pmcid_acc[year]),
               (np.array(list(dict_pmcid_acc[year].values())) ** 2).sum()) for year in uniques2]
        # normalize index to be between 0 and 1
        # nhi {year : index}
        nhi = {year: (x - 1. / n) / (1. - 1. / n) if n != 1 else 1. for year, n, x in hi}

        prev_years = dict(zip(uniques2, [-1] + list(uniques2[:-1])))
        # create an array [pmid, personal_index, nhi_index]
        for year, pmids in dict_year_pm.items():
            for pmid in pmids:
                if prev_years[year] > 0:
                    # sum of accumulated proportions for cid, that are related to current pmid
                    fracs = [dict_pmcid_acc[prev_years[year]][cid] for cid in pmids_clust_dict[pmid]
                             if cid in dict_pmcid_acc[prev_years[year]].keys()]
                    personal_frac[pmid] = (np.sum(fracs), nhi[prev_years[year]])
                else:
                    personal_frac[pmid] = (0.0, 0.0)

    result_accumulator = []
    for pmid, year in pd_incoming[[id_column, agg_column]].values:
        if personal_frac and (pmid in personal_frac.keys()):
            result_accumulator.append([pmid, *personal_frac[pmid]])
        else:
            if uniques2.size > 0:
                ind = np.argmax(year <= uniques2)
                if ind > 0:
                    best_year = uniques2[ind - 1]
                    nh_index = nhi[best_year]
                else:
                    nh_index = 0.
            else:
                nh_index = 0.
            result_accumulator.append([pmid, 0.0, nh_index])
    return pd.DataFrame(np.array(result_accumulator), index=pd_incoming.index)



def train_test_split_key(df, test_size, seed, agg_ind, stratify_key_agg, skey, verbose=False):
    pkey = stratify_key_agg
    nkey = skey
    df_key = df.drop_duplicates(agg_ind)
    df_key_train, df_key_test = train_test_split(df_key, test_size=test_size,
                                                 random_state=seed,
                                                 stratify=df_key[pkey])
    df_test = df.loc[df[agg_ind].isin(df_key_test[agg_ind].unique())]
    df_train = df.loc[df[agg_ind].isin(df_key_train[agg_ind].unique())]

    if verbose:
        print('train vc:')
        print(df_key_train[pkey].value_counts())
        print('test vc:')
        print(df_key_test[pkey].value_counts())
        print('train vc:')
        print(df_train[nkey].value_counts())
        print('test vc:')
        print(df_test[nkey].value_counts())
        print('fraction. resulting : {0:.2f}, requested {1:.2f}'
              .format(df_test.shape[0]/df.shape[0], test_size))
    return df_train, df_test
