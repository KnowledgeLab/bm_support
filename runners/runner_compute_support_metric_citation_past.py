import pandas as pd
from datahelpers.constants import iden, ye, ai, ps, up, dn, ar, ni, cexp, qcexp, nw, wi, dist, pm
from bm_support.bigraph_support import compute_support_index, compute_affinity_index
from os.path import expanduser

fraction_imporant_v_vertices = 0.2

n_test = None
# n_test = 2000

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


if n_test:
    dfy = dfy.head(n_test)

dfr = dfy.groupby([up, dn]).apply(lambda x: compute_support_index(x, cites_dict,
                                                                  pm_wid_dict, ye, 1, fraction_imporant_v_vertices))
dfr = dfr.reset_index()
print(dfr.head())
dfr.to_csv(expanduser('~/data/wos/cites/support_metric_past.csv.gz'), compression='gzip')

# concl. dfr should be merged on [up, dn, ye]

dfr2 = dfy.groupby([up, dn]).apply(lambda x: compute_affinity_index(x, cites_dict, pm_wid_dict, ye))
dfr2 = dfr2.reset_index()

dfr2 = dfr2.drop(['level_2'], axis=1)

dfr2[pm] = dfr2[pm].astype(int)
dfr2[ye] = dfr2[ye].astype(int)

print(dfr2.head())

print('affinity df shape {0}; number of non zeros {1}'.format(dfr2.shape, sum(dfr2['aff_ind'] != 0)))
dfr2.to_csv(expanduser('~/data/wos/cites/affinity_metric_past.csv.gz'), compression='gzip')
# concl. dfr should be merged on [up, dn, pm]
