import pandas as pd
from datahelpers.constants import iden, ye, ai, ps, up, dn, ar, ni, cexp, qcexp, nw, wi, dist, pm
from bm_support.bigraph_support import compute_support_index
from os.path import expanduser

df = pd.read_csv(expanduser('~/data/wos/cites/wos_citations.csv.gz'), compression='gzip', index_col=0)
df_pm_wid = pd.read_csv(expanduser('~/data/wos/cites/wosids.csv.gz'), index_col=0)
dfy = pd.read_csv(expanduser('~/data/wos/pmids/updnyearpmid_all.csv.gz'), index_col=0)

wids2analyze = df['wos_id'].unique()
df_pm_wid = df_pm_wid.loc[df_pm_wid['wos_id'].isin(wids2analyze)]

# create w2i dict
super_set = set()
for c in df.columns:
    super_set |= set(df[c].unique())
print(len(super_set))
w2i = dict(zip(list(super_set), range(len(super_set))))

df['wos_id_int'] = df['wos_id'].apply(lambda x: w2i[x])
df['uid_int'] = df['uid'].apply(lambda x: w2i[x])
df_pm_wid['wos_id_int'] = df_pm_wid['wos_id'].apply(lambda x: w2i[x])

# produce cites_dict
cites_dict = dict(df.groupby('wos_id_int').apply(lambda x: list(x['uid_int'].values)))
pm_wid_dict = dict(df_pm_wid[[pm, 'wos_id_int']].values)

# lens = dfy.groupby([up, dn, ye]).apply(lambda x: x.shape[0])
# vcs = lens.value_counts()
dfr = dfy.groupby([up, dn]).apply(lambda x: compute_support_index(x, cites_dict, pm_wid_dict, ye, 1, 0.1))

dfr.to_csv(expanduser('~/data/wos/cites/support_metric_past.csv.gz'), compression='gzip')
