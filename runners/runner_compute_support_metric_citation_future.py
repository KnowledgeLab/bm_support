import pickle
import gzip
import pandas as pd
from numpy import nan
from datahelpers.constants import iden, ye, ai, ps, up, dn, ar, ni, cexp, qcexp, nw, wi, dist, pm
from bm_support.bigraph_support import compute_support_index, compute_affinity_index
from os.path import expanduser

fraction_imporant_v_vertices = 0.2
window_size = 1

n_test = None
n_test = 2000

fname2 = expanduser('~/data/wos/cites/cites_cs_v2.pgz')
with gzip.open(fname2, 'rb') as fp:
    pack = pickle.load(fp)

# wos/uid stripped to index
w2i = pack['s2i']
cites_dict = pack['id_cited_by']

df_pm_wid = pd.read_csv(expanduser('~/data/wos/cites/wosids.csv.gz'), index_col=0)
dfy = pd.read_csv(expanduser('~/data/wos/pmids/updnyearpmid_all.csv.gz'), index_col=0)


df_pm_wid['wos_id_stripped'] = df_pm_wid['wos_id'].apply(lambda x: x[4:])

df_pm_wid['wos_id_int'] = df_pm_wid['wos_id_stripped'].apply(lambda x: w2i[x] if x in w2i.keys() else nan)
df_pm_wid = df_pm_wid[~df_pm_wid['wos_id_int'].isnull()]
df_pm_wid['wos_id_int'] = df_pm_wid['wos_id_int'].astype(int)
pm_wid_dict = dict(df_pm_wid[[pm, 'wos_id_int']].values)

outstanding = list(set(pm_wid_dict.values()) - set(cites_dict.keys()))
outstanding_dict = {k: [] for k in outstanding}
cites_dict = {**cites_dict, **outstanding_dict}

if n_test:
    dfy = dfy.head(n_test)


dfr = dfy.groupby([up, dn]).apply(lambda x: compute_support_index(x, cites_dict,
                                                                  pm_wid_dict, ye, 1,
                                                                  fraction_imporant_v_vertices, window_size))

dfr = dfr.reset_index()
print(dfr.head())

if not n_test:
    dfr.reset_index().to_csv(expanduser('~/data/wos/cites/support_metric_future.csv.gz'), compression='gzip')

# concl. dfr should be merged on [up, dn, ye]

dfr2 = dfy.groupby([up, dn]).apply(lambda x: compute_affinity_index(x, cites_dict, pm_wid_dict,
                                                                    ye, window_size))
dfr2 = dfr2.reset_index()

dfr2 = dfr2.drop(['level_2'], axis=1)

dfr2[pm] = dfr2[pm].astype(int)
dfr2[ye] = dfr2[ye].astype(int)

print(dfr2.head())

print('affinity df shape {0}; number of non zeros {1}'.format(dfr2.shape, sum(dfr2['aff_ind'] != 0)))
if not n_test:
    dfr2.to_csv(expanduser('~/data/wos/cites/affinity_metric_past.csv.gz'), compression='gzip')

# concl. dfr should be merged on [up, dn, pm]
