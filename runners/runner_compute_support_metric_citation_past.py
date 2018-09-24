import pandas as pd
from datahelpers.constants import iden, ye, ai, ps, up, dn, ar, ni, cexp, qcexp, nw, wi, dist, pm
from bm_support.bigraph_support import compute_support_index, compute_affinity_index
from os.path import expanduser

fraction_imporant_v_vertices = 0.2
window_sizes = [None, 1, 2, 3]

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

    dfr2 = dfy.groupby([up, dn]).apply(lambda x: compute_affinity_index(x, cites_dict, pm_wid_dict,
                                                                        ye, window_size))
    dfr2 = dfr2.reset_index()

    dfr2 = dfr2.drop(['level_2'], axis=1)

    dfr2[pm] = dfr2[pm].astype(int)
    dfr2[ye] = dfr2[ye].astype(int)
    dfr2 = dfr2.set_index([up, dn, ye, pm]).sort_index()

    df_agg_aff.append(dfr2)

for y, d, d2 in zip(window_sizes, df_agg_supp, df_agg_aff):
    print(y, d.shape, d2.shape)

df_agg_supp2 = pd.concat(df_agg_supp, axis=1)
print('supp concat shape: ', df_agg_supp2.shape)

df_agg_aff2 = pd.concat(df_agg_aff, axis=1)
print('aff concat shape: ', df_agg_supp2.shape)

print('Fractions of indices that are non zero:')

for c in [col for col in df_agg_supp2 if 'ind' in col]:
    print('{0} : {1:.2f} %'.format(c, 100*sum(df_agg_supp2[c] != 0)/df_agg_supp2.shape[0]))

for c in df_agg_aff2.columns:
    print('{0} : {1:.2f} %'.format(c, 100*sum(df_agg_aff2[c] != 0)/df_agg_aff2.shape[0]))

if not n_test:
    df_agg_supp2.to_csv(expanduser('~/data/wos/cites/support_metric_past.csv.gz'), compression='gzip')
else:
    print(df_agg_supp2.head())

if not n_test:
    df_agg_aff2.to_csv(expanduser('~/data/wos/cites/affinity_metric_past.csv.gz'), compression='gzip')
else:
    print(df_agg_aff2.head())
