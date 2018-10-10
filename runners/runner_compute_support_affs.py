import pickle
import gzip
import pandas as pd
from datahelpers.constants import iden, ye, ai, ps, up, dn, ar, ni, cexp, qcexp, nw, wi, dist, pm
from bm_support.bigraph_support import compute_support_index, compute_affinity_index, compute_modularity_index
from os.path import expanduser
import time

verbosity = False
fraction_imporant_v_vertices = 0.5
window_sizes = [None, 1, 2, 3]
disjoint_uv = False

support_flag = False
affinity_flag = False
# affinity_flag = True
# mod_flag = False
mod_flag = True

n_test = None
# n_test = 2000

df_pm_wid = pd.read_csv(expanduser('~/data/wos/cites/wosids.csv.gz'), index_col=0)
dfy = pd.read_csv(expanduser('~/data/wos/pmids/updnyearpmid_all.csv.gz'), index_col=0)

aff_dict_fname=expanduser('~/data/wos/affs_disambi/pm2id_dict.pgz')
if aff_dict_fname:
    with gzip.open(aff_dict_fname, 'rb') as fp:
        pm_affs_dict = pickle.load(fp)


pm_wid_dict = {}
outstanding = list(set(dfy[pm].unique()) - set(pm_affs_dict.keys()))
outstanding_dict = {k: [] for k in outstanding}
pm_affs_dict = {**pm_affs_dict, **outstanding_dict}
print('number of rows : {0}'.format(dfy.shape[0]))

if n_test:
    dfy = dfy.head(n_test)

df_agg_supp = []
df_agg_aff = []
df_agg_mod = []
times = [time.time()]

for window_size in window_sizes:
    # support
    if support_flag:
        dfr = dfy.groupby([up, dn]).apply(lambda x: compute_support_index(x, pm_affs_dict,
                                                                          pm_wid_dict, ye, window_size,
                                                                          fraction_imporant_v_vertices, use_wosids=False))

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

for y, d, d2, d3 in zip(window_sizes, df_agg_supp, df_agg_aff, df_agg_mod):
    print(y, d.shape, d2.shape, d3.shape)

if support_flag:
    df_agg_supp2 = pd.concat(df_agg_supp, axis=1)
    print('supp concat shape: ', df_agg_supp2.shape)
    print('Support: fractions of indices that are non zero:')
    for c in [col for col in df_agg_supp2 if 'ind' in col]:
        print('{0} : {1:.2f} %'.format(c, 100*sum(df_agg_supp2[c] != 0)/df_agg_supp2.shape[0]))
    if not n_test:
        df_agg_supp2.to_csv(expanduser('~/data/wos/cites/support_metric_affiliations.csv.gz'), compression='gzip')
    else:
        print(df_agg_supp2.head())

if affinity_flag:
    df_agg_aff2 = pd.concat(df_agg_aff, axis=1)
    print('aff concat shape: ', df_agg_aff2.shape)
    print('Affinity: fractions of indices that are non zero:')
    for c in df_agg_aff2.columns:
        print('{0} : {1:.2f} %'.format(c, 100 * sum(df_agg_aff2[c] != 0) / df_agg_aff2.shape[0]))
    if not n_test:
        df_agg_aff2.to_csv(expanduser('~/data/wos/cites/affinity_metric_affiliations.csv.gz'), compression='gzip')
    else:
        print(df_agg_aff2.head())

if mod_flag:
    df_agg_mod2 = pd.concat(df_agg_mod, axis=1)
    print('mod concat shape: ', df_agg_mod2.shape)
    print('Modularity: fractions of indices that are non one:')
    for c in df_agg_mod2.columns:
        print('{0} : {1:.2f} %'.format(c, 100 * sum(df_agg_mod2[c] != 1) / df_agg_mod2.shape[0]))
    if not n_test:
        df_agg_mod2.to_csv(expanduser('~/data/wos/cites/modularity_metric_affiliations.csv.gz'), compression='gzip')
    else:
        print(df_agg_mod2.head())
        df_agg_mod2.to_csv(expanduser('~/data/wos/cites/modularity_metric_affiliations.csv.gz'), compression='gzip')
