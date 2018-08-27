import pandas as pd
from datahelpers.constants import iden, ye, ai, ps, up, dn, ar, ni, cexp, qcexp, nw, wi, dist, aus, pm
from bm_support.bigraph_support import compute_support_index, compute_affinity_index
from os.path import expanduser
import gzip
import pickle

fraction_imporant_v_vertices = 0.5
window_size = 1

n_test = None
n_test = 2000

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

if n_test:
    dfy = dfy.head(n_test)

dfr = dfy.groupby([up, dn]).apply(lambda x: compute_support_index(x, pm_affs_dict,
                                                                  pm_wid_dict, ye, window_size,
                                                                  fraction_imporant_v_vertices, use_wosids=False))

dfr = dfr.reset_index()
print(dfr.head())

if not n_test:
    dfr.reset_index().to_csv(expanduser('~/data/wos/cites/support_metric_authors.csv.gz'), compression='gzip')

# concl. dfr should be merged on [up, dn, ye]

dfr2 = dfy.groupby([up, dn]).apply(lambda x: compute_affinity_index(x, pm_affs_dict, pm_wid_dict,
                                                                    ye, window_size, use_wosids=False))
dfr2 = dfr2.reset_index()

dfr2 = dfr2.drop(['level_2'], axis=1)

dfr2[pm] = dfr2[pm].astype(int)
dfr2[ye] = dfr2[ye].astype(int)

print(dfr2.head())

print('affinity df shape {0}; number of non zeros {1}'.format(dfr2.shape, sum(dfr2['aff_ind'] != 0)))
if not n_test:
    dfr2.to_csv(expanduser('~/data/wos/cites/affinity_metric_authors.csv.gz'), compression='gzip')
# concl. dfr should be merged on [up, dn, pm]
