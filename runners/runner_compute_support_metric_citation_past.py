import pandas as pd
from datahelpers.constants import iden, ye, ai, ps, up, dn, ar, ni, cexp, qcexp, nw, wi, dist, pm
from bm_support.bigraph_support import compute_support_index, compute_affinity_index, compute_modularity_index
from os.path import expanduser
import time
import random

verbosity = False
fraction_imporant_v_vertices = 0.2
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

df = pd.read_csv(expanduser('~/data/wos/cites/wos_citations.csv.gz'), compression='gzip', index_col=0)
df_pm_wid = pd.read_csv(expanduser('~/data/wos/cites/wosids.csv.gz'), index_col=0)
dfy = pd.read_csv(expanduser('~/data/wos/pmids/updnyearpmid_all.csv.gz'), index_col=0)

wids2analyze = df['wos_id'].unique()

# create w2i dict
super_set = set()
for c in df.columns:
    super_set |= set(df[c].unique())
print('len of super set', len(super_set))
w2i = dict(zip(list(super_set), range(len(super_set))))

df['wos_id_int'] = df['wos_id'].apply(lambda x: w2i[x])
df['uid_int'] = df['uid'].apply(lambda x: w2i[x])

# produce cites_dict
cites_dict = dict(df.groupby('wos_id_int').apply(lambda x: list(x['uid_int'].values)))
df_pm_wid = df_pm_wid.loc[df_pm_wid['wos_id'].isin(wids2analyze)]
df_pm_wid['wos_id_int'] = df_pm_wid['wos_id'].apply(lambda x: w2i[x])
pm_wid_dict = dict(df_pm_wid[[pm, 'wos_id_int']].values)
print('number of rows : {0}'.format(dfy.shape[0]))

if n_test:
    dfy = dfy.head(n_test)

random.seed(13)

df_agg_supp = []
df_agg_aff = []
df_agg_mod = []
df_agg_redmod = []
times = [time.time()]

for window_size in window_sizes:
    # support
    if support_flag:
        dfr = dfy.groupby([up, dn]).apply(lambda x: compute_support_index(x, cites_dict,
                                                                          pm_wid_dict, ye, window_size,
                                                                          fraction_imporant_v_vertices))
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
        dfr2 = dfy.groupby([up, dn]).apply(lambda x: compute_affinity_index(x, cites_dict, pm_wid_dict,
                                                                            ye, window_size))
        dfr2 = dfr2.reset_index()

        dfr2 = dfr2.drop(['level_2'], axis=1)

        dfr2[pm] = dfr2[pm].astype(int)
        dfr2[ye] = dfr2[ye].astype(int)
        dfr2 = dfr2.set_index([up, dn, ye, pm]).sort_index()

        df_agg_aff.append(dfr2)
        times.append(time.time())
        print('aff, window size {0} {1:.2f} sec elapsed'.format(window_size, times[-1] - times[-2]))

    # modularity
    if mod_flag:

        dfr3 = dfy.groupby([up, dn]).apply(lambda x: compute_modularity_index(x, cites_dict, pm_wid_dict,
                                                                              ye, window_size, disjoint_uv=disjoint_uv))
        dfr3 = dfr3.reset_index()

        dfr3 = dfr3.drop(['level_2'], axis=1)

        dfr3[pm] = dfr3[pm].astype(int)
        dfr3[ye] = dfr3[ye].astype(int)
        dfr3 = dfr3.set_index([up, dn, ye, pm]).sort_index()

        df_agg_mod.append(dfr3)
        times.append(time.time())
        print('mod, window size {0} {1:.2f} sec elapsed'.format(window_size, times[-1] - times[-2]))

    # modularity reduced
    if red_mod_flag:
        dfr4 = dfy.groupby([up, dn]).apply(lambda x: compute_modularity_index(x, cites_dict, pm_wid_dict,
                                                                              ye, window_size,
                                                                              use_wosids=True,
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
    df_agg_supp2 = pd.concat(df_agg_supp, axis=1)
    print('supp concat shape: ', df_agg_supp2.shape)
    print('Support: fractions of indices that are non zero:')
    for c in [col for col in df_agg_supp2 if 'ind' in col]:
        print('{0} : {1:.2f} %'.format(c, 100*sum(df_agg_supp2[c] != 0)/df_agg_supp2.shape[0]))
    if not n_test:
        df_agg_supp2.to_csv(expanduser('~/data/wos/cites/support_metric_past.csv.gz'), compression='gzip')
    else:
        print(df_agg_supp2.head())
        df_agg_supp2.to_csv(expanduser('~/data/wos/cites/support_metric_past_tmp.csv.gz'), compression='gzip')

if affinity_flag:
    df_agg_aff2 = pd.concat(df_agg_aff, axis=1)
    print('aff concat shape: ', df_agg_aff2.shape)
    print('Affinity: fractions of indices that are non zero:')
    for c in df_agg_aff2.columns:
        print('{0} : {1:.2f} %'.format(c, 100 * sum(df_agg_aff2[c] != 0) / df_agg_aff2.shape[0]))
    if not n_test:
        df_agg_aff2.to_csv(expanduser('~/data/wos/cites/affinity_metric_past.csv.gz'), compression='gzip')
    else:
        print(df_agg_aff2.head())
        df_agg_aff2.to_csv(expanduser('~/data/wos/cites/affinity_metric_past_tmp.csv.gz'), compression='gzip')

if mod_flag:
    df_agg_mod2 = pd.concat(df_agg_mod, axis=1)
    print('mod concat shape: ', df_agg_mod2.shape)
    print('Modularity: fractions of indices that are non one:')
    for c in df_agg_mod2.columns:
        print('{0} : {1:.2f} %'.format(c, 100 * sum(df_agg_mod2[c] != 1) / df_agg_mod2.shape[0]))
    if not n_test:
        df_agg_mod2.to_csv(expanduser('~/data/wos/cites/modularity_metric_past.csv.gz'), compression='gzip')
    else:
        print(df_agg_mod2.head())
        df_agg_mod2.to_csv(expanduser('~/data/wos/cites/modularity_metric_past_tmp.csv.gz'), compression='gzip')

if red_mod_flag:
    dft = pd.concat(df_agg_redmod, axis=1)
    print('mod concat shape: ', dft.shape)
    print('Modularity: fractions of indices that are non one:')
    for c in dft.columns:
        print('{0} : {1:.2f} %'.format(c, 100 * sum(dft[c] != 1) / dft.shape[0]))
    if not n_test:
        dft.to_csv(expanduser('~/data/wos/cites/redmodularity_metric_past.csv.gz'), compression='gzip')
    else:
        print(dft.head())
        dft.to_csv(expanduser('~/data/wos/cites/redmodularity_metric_past_tmp.csv.gz'), compression='gzip')
