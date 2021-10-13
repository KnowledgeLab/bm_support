import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler
from datahelpers.dftools import agg_file_info
from os import listdir
from os.path import isfile, join, expanduser
from datahelpers.constants import (
    iden,
    ye,
    ai,
    ps,
    up,
    dn,
    ar,
    cexp,
    qcexp,
    dist,
    rdist,
    pm,
    ct,
    affs,
    aus,
    bdist,
)
from datahelpers.dftools import select_appropriate_datapoints, add_column_from_file
import Levenshtein as lev
from functools import partial
from .disambiguation import ndistance, nlevenstein_root, cluster_objects
from sklearn.model_selection import train_test_split
from numpy import unique
import gzip
import pickle
from copy import deepcopy
from .derive_feature import add_t0_flag, attach_moving_averages_per_interaction
import json


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
        data_rel = (data_ - data_.mean()) / std
    if np.isnan(std):
        print(data_.shape[0], data.index, std)

    return pd.Series(data_rel, index=data.index)


def normalize_columns(df, columns, scaler=None):
    df2 = df.copy()
    if not scaler:
        scaler = MinMaxScaler()
        scaler.fit(df[columns])
    df2[columns] = scaler.transform(df[columns])
    return df2


def normalize_columns_with_scaler(df, columns, scaler=None):
    df2 = df.copy()
    if not scaler:
        scaler = MinMaxScaler()
        scaler.fit(df[columns])
    df2[columns] = scaler.transform(df[columns])
    return df2, scaler


def mask_out(df, cutoff=None, extra_masks=None, verbose=False):
    masks = []

    if extra_masks:
        masks.extend(extra_masks)

    # mask only only the upper and the lower quartiles in cdf_exp
    if cutoff:
        upper_exp, lower_exp = 1 - cutoff, cutoff
        exp_mask = [
            cexp,
            (upper_exp, lower_exp),
            lambda df_, th: (df_ >= th[0]) | (df_ <= th[1]),
        ]
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
    ps_mask = [ps, 0.0, lambda s, th: (s >= th)]
    masks.append(ps_mask)

    df_selected = select_appropriate_datapoints(df, masks)

    if verbose:
        print(
            "received: {0} rows, after masking out: {1} rows.".format(
                df.shape[0], df_selected.shape[0]
            )
        )
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
        raise ValueError("in define_distance() columns argument is not length two")
    elif not (set(columns) < set(df.columns)):
        raise ValueError("in define_distance() columns are not in df.columns")

    a, b = columns
    n_a = df[a].value_counts().shape[0] - 1
    n_b = df[b].value_counts().shape[0] - 1
    lcm_ab = lcm(n_a, n_b)
    m_a = lcm_ab / n_a
    m_b = lcm_ab / n_b
    if verbose:
        print(
            "class a scale: {0}; class b scale: {1}. lcm {2}".format(n_a, n_b, lcm_ab)
        )
        print("m a {0}; m b {1}".format(m_a, m_b))

    s = np.abs(m_b * df[b] - m_a * df[a])
    if verbose:
        print(s.value_counts(), s.mean())
    return s


def derive_distance_column(
    df,
    column_a_parameters=(cexp, qcexp, (-1.0e-8, 0.5, 1.0)),
    column_b_parameters=ps,
    distance_column="guess",
    verbose=False,
):
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


def derive_abs_pct_values(df, c):
    df[f"{c}_pct"] = df[c].rank(pct=True)
    df[f"{c}_absmed"] = (df[f"{c}_pct"] - df[f"{c}_pct"].median()).abs()
    df[f"{c}_absmed_pct"] = df[f"{c}_absmed"].rank(pct=True)
    return df


def prepare_final_df(
    df,
    normalize=False,
    columns_normalize=None,
    columns_normalize_by_interaction=None,
    cutoff=None,
    quantize_intervals=(-1.0e-8, 0.5, 1.0),
    aff_dict_fname=None,
    suppress_affs=False,
    masks=None,
    add_cite_fits=False,
    define_visible_prior=False,
    community_refutation_df=None,
    verbose=False,
):

    # mask_len_ = (df.groupby([up, dn]).apply(lambda x: x.shape[0]) > min_len)
    # mask_len = df[ni].isin(mask_len_[mask_len_].index)
    # df = df[mask_len]
    # if max_len:
    #     mask_len_ = (df.groupby(ni).apply(lambda x: x.shape[0]) < max_len)
    #     mask_len = df[ni].isin(mask_len_[mask_len_].index)
    #     df = df[mask_len]

    df_selected = mask_out(df, cutoff, masks, verbose)

    pmids = df_selected[pm].unique()

    df_wos = retrieve_wos_aff_au_df()

    if verbose:
        print("defining authors' features")
    # {pmid : [i_A]} -  dict of lists of of authors
    pm_aus_map = cluster_authors(df_wos, pmids)
    # clean {pmid : [i_A]} of possible duplicates
    pm_aus_map = {k: list(set(v)) for k, v in pm_aus_map.items()}

    # for each new_index ni (interaction) calculate authors herfindahl index
    aus_feature = df_selected.groupby([up, dn]).apply(
        lambda x: calc_normed_hi(x, pm_aus_map, (pm, ye))
    )
    aus_feature = aus_feature.rename(columns={0: pm, 1: "pre_" + aus, 2: "nhi_" + aus})
    f_cols = ["pre_" + aus, "nhi_" + aus]

    if not suppress_affs:
        if verbose:
            print("defining affiliations' features...")
        if aff_dict_fname:
            with gzip.open(aff_dict_fname, "rb") as fp:
                pm_clusters = pickle.load(fp)
            if verbose:
                print(
                    "loaded pm_clusters, dict contains {0} pmids".format(
                        len(pm_clusters)
                    )
                )
        else:
            pm_clusters = cluster_affiliations(df_wos, pmids)

        pm_clusters = {k: list(set(v)) for k, v in pm_clusters.items()}
        affs_feature = df_selected.groupby([up, dn]).apply(
            lambda x: calc_normed_hi(x, pm_clusters, (pm, ye))
        )
        affs_feature = affs_feature.rename(
            columns={0: pm, 1: "pre_" + affs, 2: "nhi_" + affs}
        )
        aus_feature = affs_feature.merge(
            aus_feature[["pre_" + aus, "nhi_" + aus]], left_index=True, right_index=True
        )
        f_cols.extend(["pre_" + affs, "nhi_" + affs])

    df_selected = df_selected.merge(
        aus_feature[f_cols], left_index=True, right_index=True
    )

    df_selected[ye + "_off"] = df_selected.groupby(
        [up, dn], as_index=False, group_keys=False
    ).apply(lambda x: x[ye] - x[ye].min())
    df_selected[ye + "_off2"] = df_selected[ye + "_off"] ** 2

    # add citation count
    fp = expanduser("~/data/literome/wos/pmid_wos_cite.csv.gz")
    dft2_ = add_column_from_file(df_selected, fp, pm, ct)

    # define distance between qcexp and ps
    dft2 = derive_distance_column(dft2_, (cexp, qcexp, quantize_intervals), ps, dist)
    dft2[rdist] = dft2[cexp] - dft2[ps]

    dft = dft2
    if verbose:
        print("value counts of distance:")
        print(dft2[dist].value_counts())

    if add_cite_fits:
        df_cites = pd.read_csv(
            expanduser("~/data/wos/cites/wos_cite_result.csv.gz"),
            compression="gzip",
            index_col=0,
        )
        dft3 = pd.merge(dft2, df_cites, on=pm, how="left")
        for c in [
            "yearspan_flag",
            "len_flag",
            "succfit_flag",
            "mu",
            "sigma",
            "A",
            "err",
            "int_3",
        ]:
            nan_mask = dft3[c].isnull()
            report_nas = sum(nan_mask)
            print(
                "{0} of {1} entries were not identified in wos db ({2:.1f}%)".format(
                    report_nas, c, 100 * report_nas / dft3.shape[0]
                )
            )
            dft3.loc[nan_mask, c] = -1.0
        report_nas = sum(dft3["sigma"].isnull())

        mask = dft3["A"] > 0
        dft3["int_3_log"] = -1.0
        dft3["A_log"] = -1.0
        dft3["int_3_log_sigma"] = -1.0
        dft3["A_log_sigma"] = -1.0

        dft3.loc[mask, "int_3_log"] = np.log(dft3.loc[mask, "int_3"] + 1)
        dft3.loc[mask, "A_log"] = np.log(dft3.loc[mask, "A"] + 1)
        dft3.loc[mask, "int_3_log_sigma"] = (
            dft3.loc[mask, "int_3_log"] - dft3.loc[mask, "int_3_log"].mean()
        ) ** 2
        dft3.loc[mask, "A_log_sigma"] = (
            dft3.loc[mask, "A_log"] - dft3.loc[mask, "A_log"].mean()
        ) ** 2
        dft = dft3
        if verbose:
            print(
                "{0} entries were not identified in wos db  ({1:.1f}%)".format(
                    report_nas, 100 * report_nas / dft3.shape[0]
                )
            )
    dft[bdist] = 1.0
    mask = 2 * dft[ps] == dft[qcexp]
    dft.loc[mask, bdist] = 0.0

    if define_visible_prior:
        mns = dft.groupby([up, dn, ye]).apply(
            lambda x: pd.Series([x[bdist].sum(), x.shape[0]], index=["mu", "s"])
        )
        mns2 = mns.groupby(level=[0, 1]).apply(
            lambda x: pd.DataFrame(
                np.array([np.cumsum(x["mu"]), np.cumsum(x["s"])]).T,
                index=x.index.get_level_values(2),
                columns=["csmu", "cssum"],
            )
        )
        mns2["mu"] = mns2["csmu"] / mns2["cssum"]
        mns3 = mns2["mu"].groupby(level=[0, 1]).apply(lambda x: x.shift())
        mns3 = mns3.reset_index().rename(columns={"mu": "obs_mu"})
        dft = dft.merge(mns3, on=[up, dn, ye])

    if normalize:
        if verbose:
            minmax = [
                "{0} min: {1:.2f}; max {2:.2f}".format(c, dft[c].min(), dft[c].max())
                for c in columns_normalize
            ]
            print(". ".join(minmax))
        if columns_normalize:
            dft = normalize_columns(dft, columns_normalize)

        for c in columns_normalize_by_interaction:
            dft[c] = dft.groupby([up, dn], as_index=False, group_keys=False).apply(
                lambda x: groupby_normalize(x[c])
            )

        if verbose:
            minmax = [
                "{0} min: {1:.2f}; max {2:.2f}".format(c, dft[c].min(), dft[c].max())
                for c in columns_normalize
            ]
            print(". ".join(minmax))

    if community_refutation_df is not None:
        dft = pd.merge(dft, community_refutation_df, how="left", on=[up, dn, pm])
    return dft


def retrieve_wos_aff_au_df(fpath="~/data/wos/wos_pmid/", verbose=False):
    fpath = expanduser(fpath)
    suffix = "txt"
    prefix = "sav"
    suffix_len = len(suffix)
    prefix_len = len(prefix)
    files = [
        f
        for f in listdir(fpath)
        if isfile(join(fpath, f))
        and (f[-suffix_len:] == suffix)
        and (f[:prefix_len] == prefix)
    ]

    if verbose:
        print(files)

    kk = ["PM", "TC", "UT", "AU", "C1"]
    ll = [agg_file_info(join(fpath, f), kk) for f in files]
    lll = [x for sublist in ll for x in sublist]

    df = pd.DataFrame(lll, columns=kk)
    df = df.rename(columns={"PM": pm, "TC": ct, "UT": "wos_id", "AU": aus, "C1": affs})
    df[pm] = df[pm].astype(int)
    return df


def process_affs(df, verbose=False):

    # number of [] () inserts in affiliations
    if verbose:
        print(
            "number of affs with authors in square brackets [] :"
            " {0}".format(sum(df[affs].apply(lambda x: "[" in x)))
        )
    # we need to exclude [abc] and (abc), they contain authors in affiliation
    # df[affs] = df[affs].apply(lambda x: re.sub('[\(\[].*?[\)\]]', '', x))
    df[affs + "_clean"] = df[affs].apply(lambda x: re.sub("[\(\[].*?[\)\]]", "", x))
    # number of empty authors, empty affiliations
    if verbose:
        print(
            "number of empty affs: {0}".format(sum(df[affs].apply(lambda x: x == "")))
        )
    df_working = df.loc[df[affs + "_clean"].apply(lambda x: x != "")]

    pm_aff_map = df_working[[pm, affs + "_clean"]].values

    pm_aff_split = [(pm_, x.split("|")) for pm_, x in pm_aff_map]

    pm_aff_phrases = []
    for pmid, phrase in pm_aff_split:
        phrase2 = [x.split(",")[0].replace("& ", "").strip().lower() for x in phrase]
        phrase2 = list(set(phrase2))
        pm_aff_phrases.append((pmid, phrase2))

    affs_list_lists = [x[1] for x in pm_aff_phrases]
    affs_lists = [x for sublist in affs_list_lists for x in sublist if x != ""]
    affs_uniq = list(set(affs_lists))

    index_aff2 = list(zip(range(len(affs_uniq)), affs_uniq))
    index_aff3 = [(j, x.split()) for j, x in index_aff2]
    a2i = dict([(aff, j) for j, aff in index_aff2])

    return index_aff3, pm_aff_phrases, a2i


def cluster_affiliations(df, pmids=[], n_processes=1, verbose=False, debug=False):

    if len(pmids) > 0:
        df = df.loc[df[pm].isin(pmids)]

    index_affs, pm_aff_phrases, a2i = process_affs(df, verbose)

    ndist = partial(ndistance, **{"foo": lev.distance})
    ndist_root = partial(nlevenstein_root, **{"foo": ndist})

    i2c, phrase_cluster_dict = cluster_objects(
        index_affs[:],
        foo=ndist_root,
        foo_basic=ndist,
        n_processes=n_processes,
        simple_thr=0.1,
        verbose=verbose,
        debug=True,
    )

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
        print(sum(df[affs].apply(lambda x: "[" in x)))

    if verbose:
        print(
            "number of empty authors: {0}",
            format(sum(df[aus].apply(lambda x: x == ""))),
        )

    df_working = df.loc[df[aus].apply(lambda x: x != "")]
    pm_aus_map = df_working[[pm, aus]].values
    pm_aus_split = [(pm_, x.split("|")) for pm_, x in pm_aus_map]

    au_llist = [x[1] for x in pm_aus_split]
    au_list = [x for sublist in au_llist for x in sublist]
    aus_unique = list(set(au_list))
    if verbose:
        print("authors :", len(au_list), ". uniques:", len(set(au_list)))

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
    ret_dict = {k: n1 * v / n for k, v in d1.items()}

    for k, v in d2.items():
        if k in ret_dict:
            ret_dict[k] += n2 * v / n
        else:
            ret_dict[k] = n2 * v / n
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
        print("u, c:", uniques, cnts)

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
            d2 = dict(zip(arr_pmids, [1.0 / length] * length))
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
            dict_pmcid_acc[q] = update_dict_numerically(
                dict_pmcid_acc[k], dict_pmcid[q], (n1, n2)
            )

        hi = [
            (
                year,
                len(dict_pmcid_acc[year]),
                (np.array(list(dict_pmcid_acc[year].values())) ** 2).sum(),
            )
            for year in uniques2
        ]
        # normalize index to be between 0 and 1
        # nhi {year : index}
        nhi = {
            year: (x - 1.0 / n) / (1.0 - 1.0 / n) if n != 1 else 1.0
            for year, n, x in hi
        }

        prev_years = dict(zip(uniques2, [-1] + list(uniques2[:-1])))
        # create an array [pmid, personal_index, nhi_index]
        for year, pmids in dict_year_pm.items():
            for pmid in pmids:
                if prev_years[year] > 0:
                    # sum of accumulated proportions for cid, that are related to current pmid
                    fracs = [
                        dict_pmcid_acc[prev_years[year]][cid]
                        for cid in pmids_clust_dict[pmid]
                        if cid in dict_pmcid_acc[prev_years[year]].keys()
                    ]
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
                    nh_index = 0.0
            else:
                nh_index = 0.0
            result_accumulator.append([pmid, 0.0, nh_index])
    return pd.DataFrame(np.array(result_accumulator), index=pd_incoming.index)


def train_test_split_key(
    df, test_size, seed, agg_ind=None, stratify_key_agg=None, skey=None, verbose=False
):
    pkey = stratify_key_agg
    nkey = skey
    if agg_ind:
        df_key = df.drop_duplicates(agg_ind)
    if pkey:
        df_key_train, df_key_test = train_test_split(
            df_key, test_size=test_size, random_state=seed, stratify=df_key[pkey]
        )
    else:
        df_key_train, df_key_test = train_test_split(
            df_key, test_size=test_size, random_state=seed
        )

    df_test = df.loc[df[agg_ind].isin(df_key_test[agg_ind].unique())]
    df_train = df.loc[df[agg_ind].isin(df_key_train[agg_ind].unique())]

    if verbose:
        if pkey:
            print("train vc:")
            print(df_key_train[pkey].value_counts())
            print("test vc:")
            print(df_key_test[pkey].value_counts())
        if nkey:
            print("train vc:")
            print(df_train[nkey].value_counts())
            print("test vc:")
            print(df_test[nkey].value_counts())
            print(
                "fraction. resulting : {0:.2f}, requested {1:.2f}".format(
                    df_test.shape[0] / df.shape[0], test_size
                )
            )

    return df_train, df_test


def generate_feature_groups(columns_filename, verbose=False):
    with open(columns_filename, "r") as f:
        line = f.read()
    columns = line.split("\n")
    columns = [x for x in columns if x != ""]

    dim1 = ["affiliations", "authors", "future", "past"]
    dim2 = ["affiliations", "authors", "future", "past", "afaupa", "afaupafu"]
    dim3 = ["affiliations", "authors", "past"]

    # rncomms, rncomponents, rcommrel, rcomm_size
    patterns = (
        ["cpop", "cden", "ksst", "nhi"]
        + ["{0}_affind".format(c) for c in dim1]
        + ["{0}_suppind".format(c) for c in dim1]
        + ["{0}_rcomm_size".format(c) for c in dim2]
        + ["{0}_rcommrel_size".format(c) for c in dim2]
        + ["{0}_rncomms".format(c) for c in dim2]
        + ["{0}_rncomponents".format(c) for c in dim2]
        + ["pos_comm_ave_{0}".format(c) for c in dim3]
        + ["pre_authors", "pre_affs"]
    )

    bipatterns = [
        ("lincs", "comm_size"),
        ("lincs", "same_comm"),
        ("litgw", "dyn_eff_comm_size"),
        ("litgw", "dyn_same_comm"),
        ("litgw", "dyn_csize_up"),
        ("litgw", "dyn_csize_dn"),
        ("litgw", "comm_size"),
        ("litgw", "same_comm"),
    ]

    if verbose:
        print("### Patterns")
        print(patterns)

    col_families = {pat: [x for x in columns if pat in x] for pat in patterns}
    col_families_prefix_suffix = {
        pat0 + pat1: [x for x in columns if pat0 in x and pat1 in x]
        for pat0, pat1 in bipatterns
    }

    # little filtering hack

    for k in ["litgwcomm_size", "litgwsame_comm"]:
        col_families_prefix_suffix[k] = [
            c for c in col_families_prefix_suffix[k] if not "dyn" in c
        ]

    col_families_basic = {k: [k] for k in ["ai", "ar", "delta_year"]}
    # fits of wos citations
    col_families["citations"] = [
        "yearspan_flag",
        "len_flag",
        "succfit_flag",
        "mu",
        "sigma",
        "cite_count",
        "A",
        "A_log",
        "A_log_sigma",
        "err",
        "int_3",
        "int_3_log",
        "int_3_log_sigma",
    ]
    col_families["time"] = ["year_off", "year_off2"]
    col_families["authors_count"] = ["authors_count"]
    col_families["affiliations_count"] = ["affiliations_count"]
    col_families["prev_rdist"] = ["prev_rdist"]
    col_families["prev_rdist_abs"] = ["prev_rdist_abs"]
    # col_families['degrees'] = ['updeg_st', 'dndeg_st', 'effdeg_st',
    #                            'updeg_end', 'dndeg_end', 'effdeg_end']
    col_families["degrees"] = [
        "degree_source_in",
        "degree_source_out",
        "degree_target_in",
        "degree_target_out",
    ]
    mu_cols = ["mu*", "mu*_pct", "mu*_absmed", "mu*_absmed_pct", "muhat"]

    col_families.update({k: [k] for k in mu_cols})

    col_families["bdist_ma"] = ["bdist_ma_None", "bdist_ma_2"]
    # col_families['obs_mu'] = ['obs_mu']

    col_families = {**col_families, **col_families_basic, **col_families_prefix_suffix}

    lens = {k: len(v) for k, v in col_families.items()}
    cols_outstanding = list(
        set(columns) - set([x for v in col_families.values() for x in v])
    )
    if verbose:
        print(lens)
        print(sum(lens.values()), len(columns))
        print(sorted(cols_outstanding)[:30])
        print(len(col_families))

    return col_families


def generate_feature_groups_coarse(columns_filename, verbose=True):
    with open(columns_filename, "r") as f:
        line = f.read()
    columns = line.split("\n")
    columns = [x for x in columns if x != ""]

    # patterns = ['cpop', 'cden', 'ksst', 'nhi'] + \
    patterns = (
        ["cden", "ksst", "nhi"]
        + ["affind"]
        + ["suppind"]
        + ["rcomm_size"]
        + ["rcommrel_size"]
        + ["rncomms"]
        + ["rncomponents"]
        + ["pos_comm_ave"]
        + ["pre_authors", "pre_affs"]
    )

    bipatterns = [
        ("lincs", "comm_size"),
        ("lincs", "same_comm"),
        ("litgw", "dyn_eff_comm_size"),
        ("litgw", "dyn_same_comm"),
        ("litgw", "dyn_csize_up"),
        ("litgw", "dyn_csize_dn"),
        ("litgw", "comm_size"),
        ("litgw", "same_comm"),
    ]

    print("### Patterns")
    print(patterns)

    col_families = {pat: [x for x in columns if pat in x] for pat in patterns}
    col_families_prefix_suffix = {
        pat0 + pat1: [x for x in columns if pat0 in x and pat1 in x]
        for pat0, pat1 in bipatterns
    }

    col_families_prefix_suffix["litgwdyn_eff_comm_size"] += col_families_prefix_suffix[
        "litgwdyn_csize_up"
    ]
    col_families_prefix_suffix["litgwdyn_eff_comm_size"] += col_families_prefix_suffix[
        "litgwdyn_csize_dn"
    ]
    del col_families_prefix_suffix["litgwdyn_csize_up"]
    del col_families_prefix_suffix["litgwdyn_csize_dn"]

    # little filtering hack

    for k in ["litgwcomm_size", "litgwsame_comm"]:
        col_families_prefix_suffix[k] = [
            c for c in col_families_prefix_suffix[k] if not "dyn" in c
        ]

    col_families_basic = {k: [k] for k in ["ai", "ar", "delta_year"]}

    # exceptional add popularity to density
    col_families["cden"] += [x for x in columns if "cpop" in x]

    # fits of wos citations
    col_families["citations"] = [
        "yearspan_flag",
        "len_flag",
        "succfit_flag",
        "mu",
        "sigma",
        "cite_count",
        "A",
        "A_log",
        "A_log_sigma",
        "err",
        "int_3",
        "int_3_log",
        "int_3_log_sigma",
    ]
    col_families["time"] = ["year_off", "year_off2"]
    col_families["authors_count"] = ["authors_count"]
    col_families["affiliations_count"] = ["affiliations_count"]
    col_families["prev_rdist"] = ["prev_rdist"]
    col_families["prev_rdist_abs"] = ["prev_rdist_abs"]
    # col_families['degrees'] = ['updeg_st', 'dndeg_st', 'effdeg_st',
    #                            'updeg_end', 'dndeg_end', 'effdeg_end']
    col_families["degrees"] = [
        "degree_source_in",
        "degree_source_out",
        "degree_target_in",
        "degree_target_out",
    ]
    mu_cols = ["mu*", "mu*_pct", "mu*_absmed", "mu*_absmed_pct", "muhat"]

    col_families.update({k: [k] for k in mu_cols})

    col_families["bdist_ma"] = ["bdist_ma_None", "bdist_ma_2"]
    # col_families['obs_mu'] = ['obs_mu']

    col_families = {**col_families, **col_families_basic, **col_families_prefix_suffix}

    lens = {k: len(v) for k, v in col_families.items()}
    cols_outstanding = list(
        set(columns) - set([x for v in col_families.values() for x in v])
    )
    if verbose:
        print(lens)
        print(sum(lens.values()), len(columns))
        print(sorted(cols_outstanding)[:30])
        print(len(col_families))

    return col_families


def select_feature_families(an_version):
    """
    returns a subset of features
    :param an_version: version
    :return:
    """

    full_families = [
        "affiliations_comm_size",
        "affiliations_ncomms",
        "affiliations_ncomponents",
        "affiliations_suppind",
        "affiliations_affind",
        "authors_comm_size",
        "authors_ncomms",
        "authors_ncomponents",
        "authors_suppind",
        "authors_affind",
        "past_comm_size",
        "past_ncomms",
        "past_ncomponents",
        "past_suppind",
        "past_affind",
        "future_comm_size",
        "future_ncomms",
        "future_ncomponents",
        "future_suppind",
        "future_affind",
        "cpop",
        "cden",
        "ksst",
        "lincscomm_size",
        "lincssame_comm",
        "litgwcomm_size",
        "litgwsame_comm",
        "ai",
        "ar",
        "citations",
        "cite_count",
        "nhi",
        "delta_year",
        "time",
    ]

    full_families_new = [
        "affiliations_comm_size",
        "affiliations_ncomms",
        "affiliations_ncomponents",
        "affiliations_suppind",
        "affiliations_affind",
        "authors_comm_size",
        "authors_ncomms",
        "authors_ncomponents",
        "authors_suppind",
        "authors_affind",
        "past_comm_size",
        "past_ncomms",
        "past_ncomponents",
        "past_suppind",
        "past_affind",
        "future_comm_size",
        "future_ncomms",
        "future_ncomponents",
        "future_suppind",
        "future_affind",
        "cpop",
        "cden",
        "ksst",
        "lincscomm_size",
        "lincssame_comm",
        "litgwcomm_size",
        "litgwsame_comm",
        "ai",
        "ar",
        "citations",
        "cite_count",
        "nhi",
        "delta_year",
        "time",
        "authors_count",
        "affiliations_count",
        "obs_mu",
    ]

    comm_families = [
        "lincscomm_size",
        "lincssame_comm",
        "litgwcomm_size",
        "litgwsame_comm",
    ]

    indep_families = [
        "affiliations_comm_size",
        "affiliations_suppind",
        "affiliations_affind",
        "authors_comm_size",
        "authors_suppind",
        "authors_affind",
        "past_comm_size",
        "past_suppind",
        "past_affind",
        "future_comm_size",
        "future_suppind",
        "future_affind",
    ]

    denindep_families = [
        "affiliations_comm_size",
        "affiliations_suppind",
        "affiliations_affind",
        "authors_comm_size",
        "authors_suppind",
        "authors_affind",
        "past_comm_size",
        "past_suppind",
        "past_affind",
        "future_comm_size",
        "future_suppind",
        "future_affind",
        "cpop",
        "cden",
        "ksst",
    ]

    nosame_nodelta_families = [
        "affiliations_comm_size",
        "affiliations_ncomms",
        "affiliations_ncomponents",
        "affiliations_suppind",
        "affiliations_affind",
        "authors_comm_size",
        "authors_ncomms",
        "authors_ncomponents",
        "authors_suppind",
        "authors_affind",
        "past_comm_size",
        "past_ncomms",
        "past_ncomponents",
        "past_suppind",
        "past_affind",
        "future_comm_size",
        "future_ncomms",
        "future_ncomponents",
        "future_suppind",
        "future_affind",
        "cpop",
        "cden",
        "ksst",
        "lincscomm_size",
        "lincssame_comm",
        "litgwcomm_size",
        "ai",
        "ar",
        "citations",
        "cite_count",
        "nhi",
        "time",
    ]

    denindep_litgw_families = [
        "litgwcomm_size",
        "affiliations_comm_size",
        "affiliations_suppind",
        "affiliations_affind",
        "authors_comm_size",
        "authors_suppind",
        "authors_affind",
        "past_comm_size",
        "past_suppind",
        "past_affind",
        "future_comm_size",
        "future_suppind",
        "future_affind",
        "cpop",
        "cden",
        "ksst",
    ]

    indiv_families = [
        "cpop",
        "cden",
        "ksst",
        "ai",
        "ar",
        "citations",
        "cite_count",
        "nhi",
        "delta_year",
        "time",
    ]

    full_nocomm = [
        "affiliations_comm_size",
        "affiliations_ncomms",
        "affiliations_ncomponents",
        "affiliations_suppind",
        "affiliations_affind",
        "authors_comm_size",
        "authors_ncomms",
        "authors_ncomponents",
        "authors_suppind",
        "authors_affind",
        "past_comm_size",
        "past_ncomms",
        "past_ncomponents",
        "past_suppind",
        "past_affind",
        "future_comm_size",
        "future_ncomms",
        "future_ncomponents",
        "future_suppind",
        "future_affind",
        "cpop",
        "cden",
        "ksst",
        "ai",
        "ar",
        "citations",
        "cite_count",
        "nhi",
        "delta_year",
        "time",
        "authors_count",
        "affiliations_count",
        "obs_mu",
    ]

    full_new = [
        "afaupa_rcomm_size",
        "afaupa_rcommrel_size",
        "afaupa_rncomms",
        "afaupa_rncomponents",
        "afaupafu_rcomm_size",
        "afaupafu_rcommrel_size",
        "afaupafu_rncomms",
        "afaupafu_rncomponents",
        "affiliations_affind",
        "affiliations_count",
        "affiliations_rcomm_size",
        "affiliations_rcommrel_size",
        "affiliations_rncomms",
        "affiliations_rncomponents",
        "affiliations_suppind",
        "ai",
        "ar",
        "authors_affind",
        "authors_count",
        "authors_rcomm_size",
        "authors_rcommrel_size",
        "authors_rncomms",
        "authors_rncomponents",
        "authors_suppind",
        "cden",
        "citations",
        "cite_count",
        "cpop",
        "delta_year",
        "ksst",
        "lincscomm_size",
        "lincssame_comm",
        "litgwcomm_size",
        "litgwdyn_eff_comm_size",
        "litgwdyn_same_comm",
        "litgwsame_comm",
        "nhi",
        "obs_mu",
        "past_affind",
        "past_rcomm_size",
        "past_rcommrel_size",
        "past_rncomms",
        "past_rncomponents",
        "past_suppind",
        "pre_affs",
        "pre_authors",
        "time",
        "pos_comm_ave_affiliations",
        "pos_comm_ave_authors",
        "pos_comm_ave_past",
        "obs_mu",
        "prev_rdist",
        "prev_rdist_abs",
    ]

    version_selector = dict()

    version_selector["full"] = full_families
    version_selector["communities"] = comm_families
    version_selector["indep"] = indep_families
    version_selector["denindep"] = denindep_families
    version_selector["nosame_nodelta"] = nosame_nodelta_families
    version_selector["denindep_litgw"] = denindep_litgw_families
    version_selector["indiv"] = indiv_families
    version_selector["nfull"] = full_families_new
    version_selector["nfull_nocomm"] = full_nocomm
    version_selector["full_version"] = full_new

    an_version_selector = {
        15: "full",
        16: "communities",
        17: "indep",
        18: "denindep",
        19: "nosame_nodelta",
        20: "denindep_litgw",
        21: "indiv",
        22: "nfull",
        23: "nfull_nocomm",
        30: "full_version",
    }

    return version_selector[an_version_selector[an_version]]


def get_mapping_data(metric_type, dfy, df_pm_wid):
    """

    :param metric_type:
    :param dfy:
    :param df_pm_wid:
    :return:
    """
    print("get_mapping_data for {0}".format(metric_type))
    if metric_type == "affiliations":
        aff_dict_fname = expanduser("~/data/wos/affs_disambi/pm2id_dict.pgz")
        if aff_dict_fname:
            with gzip.open(aff_dict_fname, "rb") as fp:
                uv_dict = pickle.load(fp)
        outstanding = list(set(dfy[pm].unique()) - set(uv_dict.keys()))
        outstanding_dict = {k: [] for k in outstanding}
        uv_dict = {**uv_dict, **outstanding_dict}
        uv_dict = {k: list(set(v)) for k, v in uv_dict.items()}

        pm_wid_dict = {}

    elif metric_type == "authors":

        df = retrieve_wos_aff_au_df()
        print("df shape {0}".format(df.shape[0]))

        df_working = df.loc[df[aus].apply(lambda x: x != "")]
        pm_aus_map = df_working[[pm, aus]].values
        pm_aus_dict = {pm_: x.lower().split("|") for pm_, x in pm_aus_map}
        print("len pm_aus_dict {0}".format(len(pm_aus_dict)))

        outstanding = list(set(dfy[pm].unique()) - set(pm_aus_dict.keys()))
        print("outstanding {0}".format(len(outstanding)))
        outstanding_dict = {k: [] for k in outstanding}
        uv_dict = {**pm_aus_dict, **outstanding_dict}
        uv_dict = {k: list(set(v)) for k, v in uv_dict.items()}

        print("uv_dict {0}".format(len(uv_dict)))

        pm_wid_dict = {}

        """
        df_working = df.loc[df[aus].apply(lambda x: x != '')]
        pm_aus_map = df_working[[pm, aus]].values
        pm_aus_dict = {pm_: x.lower().split('|') for pm_, x in pm_aus_map}
        print('len pm_aus_dict {0}'.format(len(pm_aus_dict)))
        
        outstanding = list(set(dfy[pm].unique()) - set(pm_aus_dict.keys()))
        print('outstanding {0}'.format(len(outstanding)))
        outstanding_dict = {k: [] for k in outstanding}
        pm_aus_dict = {**pm_aus_dict, **outstanding_dict}
        pm_wid_dict = {}
        
        print('len pm_wid_dict {0}'.format(len(pm_wid_dict)))
        print('len pm_aus_dict {0}'.format(len(pm_aus_dict)))        
        """

    elif metric_type == "past":
        df = pd.read_csv(
            expanduser("~/data/wos/cites/wos_citations.csv.gz"),
            compression="gzip",
            index_col=0,
        )
        wids2analyze = df["wos_id"].unique()

        # create w2i dict
        super_set = set()
        for c in df.columns:
            super_set |= set(df[c].unique())
        print("len of super set", len(super_set))
        w2i = dict(zip(list(super_set), range(len(super_set))))

        df["wos_id_int"] = df["wos_id"].apply(lambda x: w2i[x])
        df["uid_int"] = df["uid"].apply(lambda x: w2i[x])

        # produce cites_dict
        uv_dict = dict(
            df.groupby("wos_id_int").apply(lambda x: list(x["uid_int"].values))
        )

        df_pm_wid = df_pm_wid.loc[df_pm_wid["wos_id"].isin(wids2analyze)].copy()
        df_pm_wid["wos_id_int"] = df_pm_wid["wos_id"].apply(lambda x: w2i[x])
        pm_wid_dict = dict(df_pm_wid[[pm, "wos_id_int"]].values)

    elif metric_type == "future":
        fname = expanduser("~/data/wos/cites/cites_cs_v2.pgz")
        with gzip.open(fname, "rb") as fp:
            pack = pickle.load(fp)

        # wos/uid stripped to index
        w2i = pack["s2i"]
        uv_dict = pack["id_cited_by"]

        df_pm_wid["wos_id_stripped"] = df_pm_wid["wos_id"].apply(lambda x: x[4:])
        df_pm_wid["wos_id_int"] = df_pm_wid["wos_id_stripped"].apply(
            lambda x: w2i[x] if x in w2i.keys() else np.nan
        )
        df_pm_wid = df_pm_wid[~df_pm_wid["wos_id_int"].isnull()].copy()
        df_pm_wid["wos_id_int"] = df_pm_wid["wos_id_int"].astype(int)
        pm_wid_dict = dict(df_pm_wid[[pm, "wos_id_int"]].values)

        outstanding = list(set(pm_wid_dict.values()) - set(uv_dict.keys()))
        outstanding_dict = {k: [] for k in outstanding}
        uv_dict = {**uv_dict, **outstanding_dict}
    else:
        uv_dict = {}
        pm_wid_dict = {}

    return uv_dict, pm_wid_dict


def get_mapping_data_reduced(metric_type, dfy, df_pm_wid):
    """
    map wids to pmids if the mapping is present
    :param metric_type:
    :param dfy:
    :param df_pm_wid:
    :return:
    """
    uv_dict, pm_wid_dict = get_mapping_data(metric_type, dfy, df_pm_wid)
    if pm_wid_dict:
        wid_pm_map = {v: k for k, v in pm_wid_dict.items()}
        uv_dict = {
            wid_pm_map[k]: v for k, v in uv_dict.items() if k in wid_pm_map.keys()
        }

    return uv_dict


def transform_last_stage(
    df, trial_features, origin, len_thr=2, normalize=False, verbose=False
):
    masks = []
    mask_lit = (df[up] == 7157) & (df[dn] == 1026)

    for c in trial_features:
        masks.append(df[c].notnull())

    mask_notnull = pd.Series([True] * df.shape[0], index=df.index)
    for m in masks:
        mask_notnull &= m
    if verbose:
        print("Number of trial features: {0}".format(len(trial_features)))
        print(
            "Number of notnull entries (over all features): {0} from {1}".format(
                sum(mask_notnull), mask_notnull.shape
            )
        )

    if origin != "gw":
        mask_agg = mask_notnull & ~mask_lit
    else:
        mask_agg = mask_notnull

    dfw = df.loc[mask_agg].copy()
    if verbose:
        print("Number of obs_mu nulls {0}".format(sum(dfw["obs_mu"].isnull())))

    mask_len_ = dfw.groupby([up, dn]).apply(lambda x: x.shape[0]) > len_thr
    updns = mask_len_[mask_len_].reset_index()[[up, dn]]
    dfw = dfw.merge(updns, how="right", on=[up, dn])
    if trial_features and normalize:
        dfw = normalize_columns(dfw, trial_features)
    return dfw


def define_laststage_metrics(
    origin,
    predict_mode="neutral",
    datapath=None,
    thr=0,
    known_aff=False,
    top_journals=False,
    verbose=False,
):
    """

    :param origin:
    :param predict_mode: 'neutral', 'posneg', 'claims'
    :param datapath:
    :param thr:
    :param known_aff:
    :param top_journals: 
    :param verbose:
    :return:
    """
    thr_dict = {"gw": (0.218, 0.305), "lit": (0.157, 0.256)}

    feat_version = 20

    if origin == "lit":
        version = 8
    elif origin == "gw":
        version = 11
    else:
        return None

    cooked_version = 12

    an_version = 30
    excl_columns = ()

    if datapath:
        col_families = generate_feature_groups(
            expanduser(join(datapath, "v{0}_columns.txt".format(feat_version)))
        )
    else:
        col_families = generate_feature_groups(
            expanduser("~/data/kl/columns/v{0}_columns.txt".format(feat_version))
        )

    if verbose:
        print(
            "Number of col families: {0}. Keys: {1}".format(
                len(col_families), sorted(col_families.keys())
            )
        )

    col_families = {k: v for k, v in col_families.items() if "future" not in k}
    if verbose:
        print(
            "Number of col families (excl. future): {0}. Keys: {1}".format(
                len(col_families), sorted(col_families.keys())
            )
        )

    # columns_interest = [x for sublist in col_families.values() for x in sublist]
    if datapath:
        df_path = expanduser(
            join(datapath, "{0}_{1}_{2}.h5".format(origin, version, cooked_version))
        )
    else:
        df_path = expanduser(
            "~/data/kl/final/{0}_{1}_{2}.h5".format(origin, version, cooked_version)
        )
    df0 = pd.read_hdf(df_path, key="df")
    if known_aff:
        df0 = df0.loc[df0.ar > 0.0].copy()

    if top_journals:
        print(f" {origin} | {(df0.ai > 0.9).mean()}")
        df0 = df0.loc[df0.ai > 0.9].copy()

    feature_dict = deepcopy(col_families)

    families = select_feature_families(an_version)
    feature_dict = {k: v for k, v in feature_dict.items() if k in families}

    feature_dict = {
        k: list(v)
        for k, v in feature_dict.items()
        if not any([c in v for c in excl_columns])
    }

    feature_dict_inv = {}
    for k, v in feature_dict.items():
        feature_dict_inv.update({x: k for x in v})

    # define k, n for interactions -> save to df_qm
    uniq_kn = set()
    dft = df0.groupby([up, dn]).apply(
        lambda x: pd.Series(
            [sum(x[ps]), x.shape[0], x[cexp].iloc[0]], index=["k", "n", "q"]
        )
    )
    dft = dft[dft.n >= thr]

    df0 = df0.merge(dft.reset_index(), on=[up, dn])

    arr = dft[["k", "n"]].apply(lambda x: tuple(x), axis=1)
    uniq_kn |= set(arr.unique())

    # define year ymin, ymax for interactions -> df_years
    dft = df0.groupby([up, dn]).apply(
        lambda x: pd.Series([x[ye].min(), x[ye].max()], index=["ymin", "ymax"])
    )
    df_years = dft.reset_index()

    df_years = df_years.groupby([up, dn]).apply(
        lambda x: pd.Series([x.ymin.min(), x.ymax.max()], index=["ymin", "ymax"])
    )

    # load degrees per interaction
    df_degs = pd.read_csv(
        expanduser("~/data/kl/comms/interaction_network/updn_degrees_directed.csv.gz"),
        index_col=0,
    )
    # load beta distribution distances per k, n
    df_dist = pd.read_csv("~/data/kl/qmu_study/uniq_kn_dist.csv", index_col=0)
    df_curv = pd.read_csv(
        "~/data/kl/derived/gene_curvature/gene_curv_feats_all_imputed.csv.gz",
        index_col=0,
    )

    if origin == "lit":
        mask_lit = (df0[up] == 7157) & (df0[dn] == 1026)
        print("filtering out 7157-1026 from lit: {0} rows out ".format(sum(mask_lit)))
        df0 = df0.loc[~mask_lit]

    df0 = pd.merge(df0.reset_index(), df_dist, on=["k", "n"])
    df0 = pd.merge(df0, df_degs, on=[up, dn, ye])
    df0 = pd.merge(df0, df_years, on=[up, dn])
    df0 = pd.merge(df0, df_curv, on=[up, dn])
    df0["mu*"] = 1 - df0["dist"]

    # define target variables
    thr_up, thr_dn = thr_dict[origin]
    mask_pos = df0.q > (1.0 - thr_up)
    mask_neg = df0.q < thr_dn

    # define interaction
    df0["bint"] = 0.0

    if predict_mode == "neutral":
        # non neutral: 1
        df0.loc[mask_neg | mask_pos, "bint"] = 1.0
    else:
        # negative: 1
        df0.loc[mask_neg, "bint"] = 1.0
        df0 = df0.loc[mask_neg | mask_pos].copy()

        if predict_mode == "claims":

            # define claim correctness: 1 if incorrect, 0 if correct
            df0["bdist"] = 0.0
            mask_incorrect = ((df0["bint"] == 1) & (df0[ps] == 1)) | (
                (df0["bint"] == 0) & (df0[ps] == 0)
            )
            print(
                f"number of correct claims, two ways {sum(~mask_incorrect)},"
                f" {sum((df0[ps] - df0[cexp]).abs() < 0.5)}"
            )
            print(
                f"number of incorrect: {sum(mask_incorrect)}, total size: {df0.shape[0]}"
            )
            df0.loc[mask_incorrect, "bdist"] = 1.0

    if predict_mode == "neutral" or predict_mode == "posneg":
        df0 = df0.drop_duplicates([up, dn]).copy()
        df0 = derive_abs_pct_values(df0, "mu*")
    else:
        dfr = attach_moving_averages_per_interaction(df0, "bdist", ps, "bint")
        df0 = df0.merge(dfr, left_on=[up, dn, ye], right_index=True)

        dft = df0.drop_duplicates([up, dn])[[up, dn, "mu*"]].copy()
        dft = derive_abs_pct_values(dft, "mu*")

        df0 = df0.merge(
            dft[[up, dn, "mu*_pct", "mu*_absmed", "mu*_absmed_pct"]], on=[up, dn]
        )
        df0 = add_t0_flag(df0)
    return df0


def prepare_datasets(
    predict_mode_="posneg",
    thr=0,
    version_feature_groups=4,
    known_aff=False,
    top_journals=False,
    version_features=23,
):
    fname = expanduser(
        f"~/data/kl/columns/feature_groups_v{version_feature_groups}.txt"
    )
    with open(fname, "r") as f:
        feat_selector = json.load(f)

    df_dict = {}

    for origin in ["gw", "lit"]:
        df_dict[origin] = define_laststage_metrics(
            origin,
            predict_mode=predict_mode_,
            thr=thr,
            known_aff=known_aff,
            top_journals=top_journals,
            verbose=True,
        )
        print(f">>> {origin} {predict_mode_} {df_dict[origin].shape[0]}")

    if predict_mode_ == "neutral" or predict_mode_ == "posneg":
        selectors = ["interaction"]
        target_ = "bint"

        cfeatures_ = [
            # 'mu*', 'mu*_absmed',
            "mu*_pct",
            "mu*_absmed_pct",
            "degree_source",
            "degree_target",
            "curv_undir_wei",
            "curv_dir_wei",
            "curv_undir_unwei",
            "curv_dir_unwei",
            "vweight_a",
            "vweight_IN_a",
            "vweight_OUT_a",
            "vweight_b",
            "vweight_IN_b",
            "vweight_OUT_b",
        ]

        cfeatures0 = set()
        for s in selectors:
            cfeatures0 |= set(feat_selector[s])

        extra_features = [
            c
            for c in list(cfeatures0)
            if ("same" in c or "eff" in c) and ("_im_ud" in c)
        ]
        cfeatures_ += extra_features

    elif predict_mode_ == "claims":
        selectors = ["claim", "batch"]
        # selectors = ['claim']
        target_ = "bdist"

        excl_columns = ()

        col_families = generate_feature_groups(
            expanduser(f"~/data/kl/columns/v{version_features}_columns.txt")
        )

        feature_dict = deepcopy(col_families)
        feature_dict = {
            k: list(v)
            for k, v in feature_dict.items()
            if not any([c in v for c in excl_columns])
        }

        feature_dict_inv = {}

        for k, v in feature_dict.items():
            feature_dict_inv.update({x: k for x in v})

        excl_set = {"bdist_ma_None", "bdist_ma_2"}

        cfeatures0 = set()
        for s in selectors:
            cfeatures0 |= set(feat_selector[s])

        gw_excl = [c for c in list(cfeatures0) if sum(df_dict["gw"][c].isnull()) > 0]
        lit_excl = [c for c in list(cfeatures0) if sum(df_dict["lit"][c].isnull()) > 0]

        cfeatures_ = list(cfeatures0 - (set(gw_excl) | set(lit_excl) | excl_set))
    else:
        cfeatures_ = None
        target_ = None

    print("***")
    print(len(cfeatures_))

    print("***")
    print(cfeatures_)
    print("***")
    print(target_)

    return df_dict, cfeatures_, target_
