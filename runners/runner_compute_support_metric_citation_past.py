import pandas as pd
from datahelpers.constants import iden, ye, ai, ps, up, dn, ar, ni, cexp, qcexp, nw, wi, dist, pm
from bm_support.bigraph_support import compute_support_index, compute_affinity_index
from os.path import expanduser

fraction_imporant_v_vertices = 0.2
window_sizes = [None, 1, 2, 3]

n_test = None
n_test = 2000

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

df_agg_supp = []
df_agg_aff = []

for window_size in window_sizes:

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

    # concl. dfr should be merged on [up, dn, ye]

    dfr2 = dfy.groupby([up, dn]).apply(lambda x: compute_affinity_index(x, cites_dict, pm_wid_dict,
                                                                        ye, window_size))
    dfr2 = dfr2.reset_index()

    dfr2 = dfr2.drop(['level_2'], axis=1)

    dfr2[pm] = dfr2[pm].astype(int)
    dfr2[ye] = dfr2[ye].astype(int)

    ren_dict = {k: '{0}{1}'.format(k, suff) for k in dfr2.columns}
    #rename dfr2 columns with suffix
    # dfr2 = dfr2.rename(columns=ren_dict)

    df_agg_aff.append(dfr2)

df_agg_supp2 = pd.concat(df_agg_supp)


if not n_test:
    df_agg_supp2.to_csv(expanduser('~/data/wos/cites/support_metric_past.csv.gz'), compression='gzip')
else:
    print(df_agg_supp2.shape)
    print(df_agg_supp2.head())

df_agg_aff2 = pd.concat(df_agg_aff)

# print('affinity df shape {0}; number of non zeros {1}'.format(dfr2.shape, sum(dfr2['aff_ind'] != 0)))
if not n_test:
    df_agg_aff2.to_csv(expanduser('~/data/wos/cites/affinity_metric_past.csv.gz'), compression='gzip')
else:
    print(df_agg_aff2.shape)
    print(df_agg_aff2.head())
