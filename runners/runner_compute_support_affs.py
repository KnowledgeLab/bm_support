import pickle
import gzip
import pandas as pd
from datahelpers.constants import ye, up, dn, pm
from bm_support.bigraph_support import compute_support_index, compute_affinity_index, compute_modularity_index
from os.path import expanduser
import time
import random

verbosity = False
fraction_imporant_v_vertices = 0.5
# window_sizes = [None, 1, 2, 3]
window_sizes = [None]
disjoint_uv = False

support_flag = True
support_flag = False
affinity_flag = True
affinity_flag = False
mod_flag = True
mod_flag = False
red_mod_flag = True
# red_mod_flag = False

n_test = None
n_test = 2000

random.seed(13)

df_pm_wid = pd.read_csv(expanduser('~/data/wos/cites/wosids.csv.gz'), index_col=0)
dfy = pd.read_csv(expanduser('~/data/wos/pmids/updnyearpmid_all.csv.gz'), index_col=0)

aff_dict_fname = expanduser('~/data/wos/affs_disambi/pm2id_dict.pgz')
if aff_dict_fname:
    with gzip.open(aff_dict_fname, 'rb') as fp:
        pm_affs_dict = pickle.load(fp)


outstanding = list(set(dfy[pm].unique()) - set(pm_affs_dict.keys()))
outstanding_dict = {k: [] for k in outstanding}
pm_affs_dict = {**pm_affs_dict, **outstanding_dict}

pm_affs_dict = {k: list(set(v)) for k, v in pm_affs_dict.items()}

print('number of rows : {0}'.format(dfy.shape[0]))
pm_wid_dict = {}

if n_test:
    dfy = dfy.head(n_test)

df_agg_supp = []
df_agg_aff = []
df_agg_mod = []
df_agg_redmod = []

times = [time.time()]

for window_size in window_sizes:
    # support
    if support_flag:
        dfr = dfy.groupby([up, dn]).apply(lambda x: compute_support_index(x, pm_affs_dict,
                                                                          pm_wid_dict, ye, window_size,
                                                                          fraction_imporant_v_vertices,
                                                                          use_wosids=False))

        dfr = dfr.reset_index()

        if window_size:
            suff = '{0}'.format(window_size)
        else:
            suff = ''
        dfr = dfr.set_index([up, dn, ye])

        df_agg_supp.append(dfr)
        times.append(time.time())
        print('supp, window size {0} {1:.2f} sec elapsed'.format(window_size, times[-1] - times[-2]))

    # affinity
    if affinity_flag:
        dfr2 = dfy.groupby([up, dn]).apply(lambda x: compute_affinity_index(x, pm_affs_dict, pm_wid_dict,
                                                                            ye, window_size, use_wosids=False))

        dfr2 = dfr2.reset_index()

        dfr2 = dfr2.drop(['level_2'], axis=1)

        dfr2[pm] = dfr2[pm].astype(int)
        dfr2[ye] = dfr2[ye].astype(int)
        dfr2 = dfr2.set_index([up, dn, ye, pm]).sort_index()
        print(dfr2.head())
        df_agg_aff.append(dfr2)
        times.append(time.time())
        print('aff, window size {0} {1:.2f} sec elapsed'.format(window_size, times[-1] - times[-2]))

    # modularity
    if mod_flag:
        print('***')
        print(len(pm_affs_dict), len(pm_wid_dict), ye, window_size, False, disjoint_uv, dfy.shape)
        sorted_keys = sorted(pm_affs_dict.keys())

        print([(k, (len(pm_affs_dict[k]), sorted(pm_affs_dict[k])[:5])) for k in sorted_keys[-5:]])

        dfr3 = dfy.groupby([up, dn]).apply(lambda x: compute_modularity_index(x, pm_affs_dict, pm_wid_dict,
                                                                              ye, window_size,
                                                                              use_wosids=False,
                                                                              disjoint_uv=disjoint_uv,
                                                                              verbose=verbosity))
        print(dfr3.shape)
        dfr3 = dfr3.reset_index()

        dfr3 = dfr3.drop(['level_2'], axis=1)

        dfr3[pm] = dfr3[pm].astype(int)
        dfr3[ye] = dfr3[ye].astype(int)
        dfr3 = dfr3.set_index([up, dn, ye, pm]).sort_index()
        print(dfr3.head())
        df_agg_mod.append(dfr3)
        times.append(time.time())
        print('mod, window size {0} {1:.2f} sec elapsed'.format(window_size, times[-1] - times[-2]))

    # modularity reduced
    if red_mod_flag:
        dfr4 = dfy.groupby([up, dn]).apply(lambda x: compute_modularity_index(x, pm_affs_dict, pm_wid_dict,
                                                                              ye, window_size,
                                                                              use_wosids=False,
                                                                              disjoint_uv=disjoint_uv,
                                                                              modularity_mode='u',
                                                                              verbose=verbosity))
        print(dfr4.shape)
        dfr4 = dfr4.reset_index()

        dfr4 = dfr4.drop(['level_2'], axis=1)

        dfr4[pm] = dfr4[pm].astype(int)
        dfr4[ye] = dfr4[ye].astype(int)
        dfr4 = dfr4.set_index([up, dn, ye, pm]).sort_index()
        print(dfr4.head())
        df_agg_redmod.append(dfr4)
        times.append(time.time())
        print('mod, window size {0} {1:.2f} sec elapsed'.format(window_size, times[-1] - times[-2]))

for y, d, d2, d3 in zip(window_sizes, df_agg_supp, df_agg_aff, df_agg_mod):
    print(y, d.shape, d2.shape, d3.shape)

if support_flag:
    dft = pd.concat(df_agg_supp, axis=1)
    print('supp concat shape: ', dft.shape)
    print('Support: fractions of indices that are non zero:')
    for c in [col for col in dft if 'ind' in col]:
        print('{0} : {1:.2f} %'.format(c, 100 * sum(dft[c] != 0) / dft.shape[0]))
    if not n_test:
        dft.to_csv(expanduser('~/data/wos/cites/support_metric_affiliations.csv.gz'), compression='gzip')
    else:
        print(dft.head())
        dft.to_csv(expanduser('~/data/wos/cites/support_metric_affiliations_tmp.csv.gz'), compression='gzip')

if affinity_flag:
    dft = pd.concat(df_agg_aff, axis=1)
    print('aff concat shape: ', dft.shape)
    print('Affinity: fractions of indices that are non zero:')
    for c in dft.columns:
        print('{0} : {1:.2f} %'.format(c, 100 * sum(dft[c] != 0) / dft.shape[0]))
    if not n_test:
        dft.to_csv(expanduser('~/data/wos/cites/affinity_metric_affiliations.csv.gz'), compression='gzip')
    else:
        print(dft.head())
        dft.to_csv(expanduser('~/data/wos/cites/affinity_metric_affiliations_tmp.csv.gz'), compression='gzip')

if mod_flag:
    dft = pd.concat(df_agg_mod, axis=1)
    print('mod concat shape: ', dft.shape)
    print('Modularity: fractions of indices that are non one:')
    for c in dft.columns:
        print('{0} : {1:.2f} %'.format(c, 100 * sum(dft[c] != 1) / dft.shape[0]))
    if not n_test:
        dft.to_csv(expanduser('~/data/wos/cites/modularity_metric_affiliations.csv.gz'), compression='gzip')
    else:
        print(dft.head())
        dft.to_csv(expanduser('~/data/wos/cites/modularity_metric_affiliations_tmp.csv.gz'), compression='gzip')

if red_mod_flag:
    dft = pd.concat(df_agg_redmod, axis=1)
    print('mod concat shape: ', dft.shape)
    print('Modularity: fractions of indices that are non one:')
    for c in dft.columns:
        print('{0} : {1:.2f} %'.format(c, 100 * sum(dft[c] != 1) / dft.shape[0]))
    if not n_test:
        dft.to_csv(expanduser('~/data/wos/cites/redmodularity_metric_affiliations.csv.gz'), compression='gzip')
    else:
        print(dft.head())
        dft.to_csv(expanduser('~/data/wos/cites/redmodularity_metric_affiliations_tmp.csv.gz'), compression='gzip')
