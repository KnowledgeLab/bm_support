from copy import deepcopy
import numpy as np
from datahelpers.constants import iden, ye, ai, ps, up, dn
from numpy.random import RandomState
from sklearn.linear_model import LinearRegression
from collections import Iterable
import pandas as pd


def sample_from_heap_dict(heap_dict, invfoo, rns, inv_foo_params, frac_test=[0.5, 0.5]):
    if isinstance(frac_test, Iterable):
        frac_test = np.array(frac_test)/np.sum(frac_test)

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
    total_cnt = sum([len(v)*k for k, v in heap_dict_working.items()])
    if isinstance(frac_test, Iterable):
        r = [get_frac(heap_dict_working, f, total_cnt) for f in frac_test[1:]]
        return [heap_dict_working] + r
    else:
        r = get_frac(heap_dict_working, frac_test, total_cnt)
        return heap_dict_working, r


def pdf(x, norm, beta):
    return norm * x ** beta


def inv_cdf(y, norm, beta, xmin):
    return (y * (beta + 1) / norm + xmin ** (beta + 1)) ** (1. / (beta + 1))


def sample_by_length(df, agg_columns=(up, dn), head=10, seed=11, frac_test=0.4, verbose=False):
    counts = df.groupby(agg_columns).apply(lambda x: x.shape[0])

    vcs = counts.value_counts()
    # we assume counts have a power law distribution
    xs = np.log(np.array(vcs.index))
    ys = np.log(vcs.values)
    xmin, xmax = counts.min(), counts.max()
    xa, xb = xmin - 0.5, xmax + 0.5
    reg = LinearRegression().fit(xs[:head].reshape(-1, 1), ys[:head])
    beta = reg.coef_[0]
    if verbose:
        print('power law exponent: {0:.3f}'.format(beta))
    norm_inv = (xb ** (beta + 1) - xa ** (beta + 1)) / (beta + 1)
    norm = 1./norm_inv

    # y = cdf(x) = A (x**(beta+1) - a**(beta+1))/(beta+1)
    # (y*(beta+1)/A + a**(beta+1))**(1./(beta+1)) = x
    rns = RandomState(seed)

    heap_dict = {}
    for ii, item in counts.iteritems():
        if item in heap_dict.keys():
            heap_dict[item].append(ii)
        else:
            heap_dict[item] = [ii]

    heap_dict = {k: sorted(v) for k, v in heap_dict.items()}
    kwargs = {'norm': norm, 'beta': beta, 'xmin': xa}
    if isinstance(frac_test, Iterable):
        dict_kfold = sample_from_heap_dict(heap_dict, inv_cdf, rns, kwargs, frac_test)
        keys_fold = [pd.DataFrame([x for sublist in dd.values() for x in sublist], columns=agg_columns)
                     for dd in dict_kfold]
        dfs = [df.merge(keys, how='inner', on=agg_columns) for keys in keys_fold]
        return dfs
    else:

        dict_train, dict_test = sample_from_heap_dict(heap_dict, inv_cdf, rns, kwargs, frac_test)
        if verbose:
            total_cnt = sum([len(v) * k for k, v in dict_train.items()])
            total_cnt2 = sum([len(v) * k for k, v in dict_test.items()])
            print('total size of train and test : {0} {1}'.format(total_cnt, total_cnt2))
            print('Ratio test to all: {0}. Should be {1}'.format(total_cnt2/(total_cnt + total_cnt2), frac_test))
        keys_train = pd.DataFrame([x for sublist in dict_train.values() for x in sublist], columns=agg_columns)
        keys_test = pd.DataFrame([x for sublist in dict_test.values() for x in sublist], columns=agg_columns)

        df_train = df.merge(keys_train, how='inner', on=agg_columns)
        df_test = df.merge(keys_test, how='inner', on=agg_columns)
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
                print('df_train power law exponent: {0:.3f}'.format(beta_train))
                print('df_test power law exponent: {0:.3f}'.format(beta_test))

        return df_train, df_test
