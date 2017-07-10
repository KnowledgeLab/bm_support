import pandas as pd
from itertools import product
import numpy as np
import pickle
import gzip
from scipy.stats import norm
from os import listdir
from os.path import isfile, join, expanduser


iden = 'identity'
ye = 'year'
ai = 'ai'
hi_ai = 'aihi'
ps = 'pos'

feauture_cols = [ai, hi_ai]
data_columns = [ye, iden] + feauture_cols + [ps]

up = 'up'
dn = 'dn'
version = 8
origin = 'gw'
data_cols = '_'.join(data_columns)
batchsize = 1000
a = 0.1
b = 0.9
n = 20
o_columns = [up, dn]
func = 'model_e'
case = 'a'


def get_id_up_dn_df(origin, n, a, b, version):
    # load id up dn freq DataFrame
    prefix_str = 'pairs_freq_{0}_v_{1}_n_{2}_a_{3}_b_{4}.csv.gz'.format(origin, version, n, a, b)
    df = pd.read_csv(join(expanduser('~/data/kl/claims/'), prefix_str),
                     compression='gzip', index_col=2)
    return df


def get_reports(origin, version, datatype, batchsize, n, a, b, func, case):
    # load available reports
    fpath = expanduser('~/data/kl/reports')

    prefix_str = '{0}_v_{1}_c_{2}_m_{3}_n_{4}_a_{5}_' \
                 'b_{6}_f_{7}_case_{8}'.format(origin, version, datatype, batchsize,
                                               n, a, b, func, case)

    files = [f for f in listdir(fpath) if isfile(join(fpath, f)) and
             (prefix_str in f)]

    reports = {}

    for f in files:
        with gzip.open(join(fpath, f), 'rb') as fp:
            rep = pickle.load(fp)
        reports.update(rep)
    reports_df = pd.DataFrame.from_dict(reports).T
    reports_df.index = map(int, reports_df.index)
    return reports_df


def get_up_dn_report(origin, version, datatype, batchsize, n, a, b, func, case):
    # attach [up, dn] to reports
    df_pairs = get_id_up_dn_df(origin, n, a, b, version)
    reports_df = get_reports(origin, version, datatype, batchsize, n, a, b, func, case)
    df_merged = pd.merge(df_pairs[o_columns], reports_df, right_index=True, left_index=True, how='right')
    return df_merged


def get_lincs_df(origin, version, n, a, b):
    # load lincs data
    fpath = expanduser('~/data/kl/claims')
    df = pd.read_csv(join(fpath, 'lincs_{0}_v_{1}_n_{2}_a_{3}_b_{4}.csv.gz'.format(origin, version, n, a, b)),
                     compression='gzip')
    return df


def process(args_reports_list, args_lincs_list):
    results = []

    for args, dfl in args_lincs_list:
        # which reports cases align with current lincs?
        reps = list(filter(lambda x: args.items() <= x[0].items(), args_reports_list))

        ccs = ['pert_itime', 'cell_id', 'pert_idose', 'pert_type', 'is_touchstone']
        cnts = [6, 5, 4, 4, 2]
        acc = []
        for c, i in zip(ccs, cnts):
            vc = dfl[c].value_counts()
            suffix = list(vc.iloc[:i].index)
            acc.append(suffix)

        # combos = list(product(*acc))

        #         acc2 = (('96 h', '120 h', '24 h', '1 h', '4 h'), ('MCF7', 'PC3', 'A375', 'VCAP', 'HA1E'),
        #                 ('2 µL', '1 µL', '1.5 µL', '100 ng/µL'), ('trt_sh', 'trt_lig', 'trt_sh.cgs', 'trt_oe'), (1, 0))
        acc2 = (('96 h', '120 h', '24 h', '1 h', '4 h'), ('MCF7', 'PC3', 'A375', 'VCAP', 'HA1E'),
                ('2 µL', '1 µL', '1.5 µL', '100 ng/µL'), [('trt_oe')], (1, 0))

        combos2 = list(product(*acc2))

        for combo in combos2:
            mask_acc = pd.Series(np.array([1] * dfl.shape[0]), index=dfl.index)
            for c, k in zip(ccs, combo):
                mask = (dfl[c] == k)
                mask_acc &= mask
            s = sum(mask_acc)
            if s > 10:
                dfl2 = dfl.loc[mask_acc]
                dfl3 = dfl2.groupby([up, dn, 'pert_type', 'cell_id',
                                     'pert_idose', 'pert_itime',
                                     'is_touchstone']).apply(lambda x: x['score'].mean()).reset_index()
                for args, df in reps:
                    df_cmp = pd.merge(df[[up, dn, 'freq', 'pi_last']], dfl3[[up, dn, 0]],
                                      how='inner', on=o_columns)
                    x = df_cmp[0].values
                    xp = list(map(lambda l: norm.cdf(l), x))
                    y = df_cmp['freq'].values
                    z = df_cmp['pi_last'].values
                    size = df_cmp.shape[0]

                    cov_freq_ = np.corrcoef(x, y)[0, 1]
                    cov_flat_ = np.corrcoef(x, z)[0, 1]

                    res_dict = {}
                    res_dict.update(args)
                    res_dict.update(dict(zip(['pert_itime', 'cell_id',
                                              'pert_idose', 'pert_type', 'is_touchstone'], combo)))
                    res_dict.update(dict(zip(['size', 'cov_freq', 'cov_model'], [s, cov_freq_, cov_flat_])))
                    results.append(res_dict)
    return results

versions = [8]
cases = ['a', 'b']
keys = ('version', 'case')
invariant_args = {
    'origin': origin,
    'datatype': data_cols,
    'batchsize': batchsize,
    'n': n,
    'a': a,
    'b': b,
    'func': func}

largs = [{k: v for k, v in zip(keys, p)} for p in product(*(versions, cases))]
full_largs = [{**invariant_args, **dd} for dd in largs]
dfs = [get_up_dn_report(**dd) for dd in full_largs]
reports_list = list(zip(full_largs, dfs))

versions = [8]
keys = ['version']

invariant_args = {
    'origin': origin,
    'n': n,
    'a': a,
    'b': b
}

largs = [{k: v for k, v in zip(keys, p)} for p in product(*[versions])]
full_largs = [{**invariant_args, **dd} for dd in largs]

dfls = [get_lincs_df(**dd) for dd in full_largs]
list(zip(full_largs, [x.shape for x in dfls]))
lincs_list = list(zip(full_largs, dfls))

rr = process(reports_list, lincs_list)

out_path = expanduser('~/data/kl/corrs/')

dfr = pd.DataFrame.from_dict(rr)
dfr.to_csv(join(out_path, 'corrs.csv'), index=False, float_format='%.5f')
