import pickle
import gzip
import pandas as pd
from datahelpers.constants import iden, ye, ai, ps, up, dn, ar, ni, cexp, qcexp, nw, wi, dist, pm
from bm_support.add_features import compute_support_index
from os.path import expanduser


fname2 = expanduser('~/data/wos/cites/cites_cs_v2.pgz')
with gzip.open(fname2, 'rb') as fp:
    pack = pickle.load(fp)

# wos/uid stripped to index
w2i = pack['s2i']
cites_dict = pack['id_cited_by']

df_pm_wid = pd.read_csv(expanduser('~/data/wos/cites/wosids.csv.gz'), index_col=0)
dfy = pd.read_csv(expanduser('~/data/wos/pmids/updnyearpmid_all.csv.gz'), index_col=0)


df_pm_wid['wos_id_stripped'] = df_pm_wid['wos_id'].apply(lambda x: x[4:])

df_pm_wid['wos_id_int'] = df_pm_wid['wos_id_stripped'].apply(lambda x: w2i[x] if x in w2i.keys() else np.nan)
df_pm_wid = df_pm_wid[~df_pm_wid['wos_id_int'].isnull()]
df_pm_wid['wos_id_int'] = df_pm_wid['wos_id_int'].astype(int)
pm_wid_dict = dict(df_pm_wid[[pm, 'wos_id_int']].values)

dfr = dfy.groupby([up, dn]).apply(lambda x: compute_support_index(x, cites_dict, pm_wid_dict, ye, 1, 0.1))
