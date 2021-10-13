from copy import deepcopy
import numpy as np
from datahelpers.constants import iden, ye, ai, ps, up, dn, bdist, large_int
from numpy.random import RandomState
from sklearn.linear_model import LinearRegression
from collections import Iterable
import pandas as pd
from .add_features import derive_abs_pct_values
from scipy.optimize import bisect


def sample_from_heap_dict(
    heap_dict, invfoo, rns, inv_foo_params, frac_test=(0.5, 0.5), verbose=False
):
    """

    :param heap_dict:
    :param invfoo:
    :param rns:
    :param inv_foo_params:
    :param frac_test:
    :param verbose:
    :return:
    """

    if isinstance(frac_test, Iterable):
        frac_test = np.array(frac_test) / np.sum(frac_test)

    if verbose:
        print(frac_test)

    def get_frac(hdict_working, frac, total_count):
        new_heap_dict = {}

        new_cnt = 0
        while new_cnt / total_count < frac:
            rv = rns.uniform()
            rv_len = invfoo(rv, **inv_foo_params)
            discreet_rvs = np.array(sorted(hdict_working.keys()))
            diffs = np.abs(discreet_rvs - rv_len)
            ii = np.argmin(diffs)
            kk = discreet_rvs[ii]
            npop = rns.randint(len(hdict_working[kk]))
            x = hdict_working[kk].pop(npop)
            if not hdict_working[kk]:
                del hdict_working[kk]
            if kk in new_heap_dict.keys():
                new_heap_dict[kk].append(x)
            else:
                new_heap_dict[kk] = [x]
            new_cnt = sum([len(v) * k for k, v in new_heap_dict.items()])
        return new_heap_dict

    heap_dict_working = deepcopy(heap_dict)
    total_cnt = sum([len(v) * k for k, v in heap_dict_working.items()])
    total_cnt_int = sum([len(v) for k, v in heap_dict_working.items()])
    if verbose:
        print(f"{total_cnt} of {total_cnt_int}")

    if isinstance(frac_test, Iterable):
        r = [get_frac(heap_dict_working, f, total_cnt) for f in frac_test[1:]]
        ret = [heap_dict_working] + r
        if verbose:
            print(
                f"total count {[sum([len(v)*k for k, v in tmp.items()]) for tmp in ret]}"
            )
            print(
                f"interaction count {[sum([len(v) for k, v in tmp.items()]) for tmp in ret]}"
            )

        return ret
    else:
        r = get_frac(heap_dict_working, frac_test, total_cnt)
        return heap_dict_working, r


def pdf(x, norm, beta):
    return norm * x ** beta


def inv_cdf(y, norm, beta, xmin):
    return (y * (beta + 1) / norm + xmin ** (beta + 1)) ** (1.0 / (beta + 1))


def sample_by_length(
    df,
    rns,
    agg_columns=(up, dn),
    head=10,
    frac_test=0.4,
    target_name=bdist,
    len_column=None,
    verbose=False,
):
    if len_column:
        counts = df.groupby(list(agg_columns)).apply(lambda x: x[len_column].iloc[0])
    else:
        counts = df.groupby(list(agg_columns)).apply(lambda x: x.shape[0])

    vcs = counts.value_counts()
    # we assume counts have a power law distribution
    xs = np.log(np.array(vcs.index))
    ys = np.log(vcs.values)
    xmin, xmax = counts.min(), counts.max()
    xa, xb = xmin - 0.5, xmax + 0.5
    reg = LinearRegression().fit(xs[:head].reshape(-1, 1), ys[:head])
    beta = reg.coef_[0]
    if verbose:
        print("power law exponent: {0:.3f}".format(beta))
    norm_inv = (xb ** (beta + 1) - xa ** (beta + 1)) / (beta + 1)
    norm = 1.0 / norm_inv

    # y = cdf(x) = A (x**(beta+1) - a**(beta+1))/(beta+1)
    # (y*(beta+1)/A + a**(beta+1))**(1./(beta+1)) = x

    # {cnt: [(id_a, id_b), ...]}
    heap_dict = {}
    for ii, item in counts.iteritems():
        if item in heap_dict.keys():
            heap_dict[item].append(ii)
        else:
            heap_dict[item] = [ii]

    heap_dict = {k: sorted(v) for k, v in heap_dict.items()}

    kwargs = {"norm": norm, "beta": beta, "xmin": xa}
    if isinstance(frac_test, Iterable):
        dict_kfold = sample_from_heap_dict(
            heap_dict, inv_cdf, rns, kwargs, frac_test, verbose=verbose
        )
        keys_fold = [
            pd.DataFrame(
                [x for sublist in dd.values() for x in sublist], columns=agg_columns
            )
            for dd in dict_kfold
        ]
        dfs = [df.merge(keys, how="inner", on=agg_columns) for keys in keys_fold]
        powers = [df[target_name].unique().shape[0] for df in dfs]
        flag_good_strat = all([p == 2 for p in powers])
        return dfs, flag_good_strat
    else:

        dict_train, dict_test = sample_from_heap_dict(
            heap_dict, inv_cdf, rns, kwargs, frac_test, verbose=verbose
        )
        if verbose:
            total_cnt = sum([len(v) * k for k, v in dict_train.items()])
            total_cnt2 = sum([len(v) * k for k, v in dict_test.items()])
            print(
                "total size of train and test : {0} {1}".format(total_cnt, total_cnt2)
            )
            print(
                "Ratio test to all: {0}. Should be {1}".format(
                    total_cnt2 / (total_cnt + total_cnt2), frac_test
                )
            )
        keys_train = pd.DataFrame(
            [x for sublist in dict_train.values() for x in sublist], columns=agg_columns
        )
        keys_test = pd.DataFrame(
            [x for sublist in dict_test.values() for x in sublist], columns=agg_columns
        )

        df_train = df.merge(keys_train, how="inner", on=agg_columns)
        df_test = df.merge(keys_test, how="inner", on=agg_columns)
        if verbose:
            counts = df_train.groupby(agg_columns).apply(lambda x: x.shape[0])
            vcs = counts.value_counts()
            xs = np.log(np.array(vcs.index))
            ys = np.log(vcs.values)
            reg = LinearRegression().fit(xs[:head].reshape(-1, 1), ys[:head])
            beta_train = reg.coef_[0]

            counts = df_test.groupby(agg_columns).apply(lambda x: x.shape[0])
            vcs = counts.value_counts()
            xs = np.log(np.array(vcs.index))
            ys = np.log(vcs.values)
            reg = LinearRegression().fit(xs[:head].reshape(-1, 1), ys[:head])
            beta_test = reg.coef_[0]
            if verbose:
                print("df_train power law exponent: {0:.3f}".format(beta_train))
                print("df_test power law exponent: {0:.3f}".format(beta_test))

        return df_train, df_test


def yield_splits(
    dfs_dict,
    rns,
    len_thr=0,
    n_splits=3,
    len_column=None,
    rank_mustar=True,
    target="bdist",
    verbose=False,
):
    df_kfolds = {}
    for k, df0 in dfs_dict.items():
        if not isinstance(len_thr, tuple) and len_column:
            df2 = df0[df0[len_column] > len_thr].copy()
        else:
            df2 = df0.copy()
        df_kfolds[k] = []

        pathology_flag = True
        while pathology_flag:
            dfs, flag = sample_by_length(
                df2,
                rns,
                (up, dn),
                5,
                [1] * n_splits,
                len_column=len_column,
                verbose=verbose,
            )

            vcs = [df_[target].unique().shape[0] for df_ in dfs]
            if verbose:
                print(vcs)
            pathology_flag = any([v == 1 for v in vcs])

        for j in range(n_splits):
            dtrain = pd.concat(dfs[:j] + dfs[j + 1 :], axis=0)
            dtest = dfs[j].copy()

            if isinstance(len_thr, tuple):
                dtrain = dtrain[dtrain.n > len_thr[0]]
                dtest = dtest[dtest.n > len_thr[1]]
            if rank_mustar:
                dtrain = derive_abs_pct_values(dtrain, "mu*")
                dtest = derive_abs_pct_values(dtest, "mu*")
            df_kfolds[k].append((dtrain, dtest))
    return df_kfolds


def yield_splits_plain(
    df0,
    rns,
    n_splits=3,
    len_column=None,
    len_thr=0,
    rank_mustar=True,
    target="bdist",
    verbose=False,
):
    df_kfolds = []
    if not isinstance(len_thr, tuple) and len_column:
        df2 = df0[df0[len_column] > len_thr].copy()
    else:
        df2 = df0.copy()

    pathology_flag = True
    while pathology_flag:
        dfs, flag = sample_by_length(
            df2,
            rns,
            agg_columns=(up, dn),
            head=10,
            frac_test=[1] * n_splits,
            len_column=len_column,
            verbose=verbose,
        )

        vcs = [df_[target].unique().shape[0] for df_ in dfs]

        int_sizes = [df_.drop_duplicates([up, dn]).shape[0] for df_ in dfs]
        tsize = sum(int_sizes)
        pathology_flag = any([x / tsize < 0.2 / n_splits for x in int_sizes])
        if verbose:
            print(vcs)
        pathology_flag |= any([v == 1 for v in vcs])

    for j in range(n_splits):
        dtrain = pd.concat(dfs[:j] + dfs[j + 1 :], axis=0)
        dtest = dfs[j].copy()

        if isinstance(len_thr, tuple):
            dtrain = dtrain[dtrain.n > len_thr[0]]
            dtest = dtest[dtest.n > len_thr[1]]
        if rank_mustar:
            dtrain = derive_abs_pct_values(dtrain, "mu*")
            dtest = derive_abs_pct_values(dtest, "mu*")
        df_kfolds.append((dtrain, dtest))
    return df_kfolds


def fill_seqs(
    pdf_dict_imperfect,
    pdf_dict_perfect,
    n_projects=100,
    wcolumn="accounted",
    direction=min,
    rns=13,
):
    """

    flag_df is of form:
        up    dn    year    size    accounted
        1543  2944  2008    2       True
        1543  2944  2009    2       False

    distribute projects for n_projects
    to fill the flags up in pdf_dict_imperfect

    :param pdf_dict_imperfect: dict of underfilled flag_dfs
    :param pdf_dict_perfect: list of filled flag_dfs
    :param n_projects: number of projects to till
    :param wcolumn: flag column
    :param direction: min or max
    :param rns:
    :return:
    """
    cnt = n_projects
    while cnt > 0 and pdf_dict_imperfect:
        if direction != "random":
            extremity_filled = direction(pdf_dict_imperfect.keys())
        else:
            skeys = sorted(pdf_dict_imperfect.keys())
            lens = np.array([len(pdf_dict_imperfect[k]) for k in skeys])
            probs = lens / lens.sum()
            if not isinstance(rns, RandomState):
                rns = RandomState(rns)
            extremity_filled = rns.choice(skeys, p=probs)
        cdf = pdf_dict_imperfect[extremity_filled].pop()
        if not pdf_dict_imperfect[extremity_filled]:
            del pdf_dict_imperfect[extremity_filled]
        ye_next = cdf.loc[~cdf[wcolumn], ye].iloc[0]
        delta = cdf.loc[~cdf[wcolumn], "size"].iloc[0]
        cdf[wcolumn] = cdf[ye] <= ye_next + 1e-6
        if cdf[wcolumn].all():
            pdf_dict_perfect.append(cdf)
        else:
            n_filled = sum(cdf.loc[cdf[wcolumn], "size"])
            if n_filled in pdf_dict_imperfect.keys():
                pdf_dict_imperfect[n_filled] += [cdf]
            else:
                pdf_dict_imperfect[n_filled] = [cdf]
        cnt -= delta
    return pdf_dict_imperfect, pdf_dict_perfect


def assign_chrono_flag(group, wcolumn="accounted", column_time=ye, frac=0.5):
    """"
    record the flag wcolumn for a fraction frac for column column_time

    :param group: Dataframe used in group by
    :param wcolumn: column name to record the flag
    :param column_time: column with which we sort
    :param frac:
    :return:

    csize ~ cumulative size
    """

    if len(group[column_time].unique()) > 1:
        group2 = group
        group2["csize"] = np.cumsum(group2.sort_values(column_time)["size"])
        cumsums = [0] + list(group2["csize"].values)
        cdf_level = frac * group2["size"].sum()
        size = group2.shape[0]

        def foo(x):
            x = int(np.round(x))
            return cumsums[x] - cdf_level

        index = min([int(np.floor(bisect(foo, 0, size, xtol=0.5))), size - 1])
        time_level = group2.iloc[index][column_time]
        group2[wcolumn] = False
        group2.loc[group2[column_time] < time_level + 1e-6, wcolumn] = True
        return group2
    else:
        group["csize"] = group["size"]
        group[wcolumn] = True
        return group
