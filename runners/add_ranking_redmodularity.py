from datahelpers.constants import ye, up, dn, pm
from bm_support.basic import rank_degenerate_values
from functools import reduce
from os.path import expanduser
import pandas as pd

head = -1
# head = 1000
metric_sources = ['authors', 'affiliations', 'future', 'past', 'afaupa', 'afaupafu']
# metric_sources = ['authors']
fpath = '~/data/wos/comm_metrics/'
windows = [1, 2, 3, None]
id_col0, relsize_col0 = 'rcommid', 'rcommrel_size'

mt = 'redmodularity'

for ms in metric_sources:
    merge_cols = [up, dn, ye, pm]
    df_agg = []
    for w in windows:
        print(f' *** : {ms} {w}')
        df = pd.read_csv(expanduser(f'{fpath}{mt}_metric_{ms}_w{w}.csv.gz'), index_col=0)

        if w:
            suff = f'{w}'
        else:
            suff = ''

        id_col = f'{id_col0}{suff}'
        relsize_col = f'{relsize_col0}{suff}'
        rank_col = f'rank{suff}'

        if head > 0:
            df = df.head(head)
        df2 = df.groupby([up, dn, 'ylook'], as_index=False, group_keys=False).apply(lambda item:
                                item.drop_duplicates(id_col)[[up, dn, 'ylook', relsize_col, id_col]])

        df3 = df2.groupby([up, dn, 'ylook']).apply(lambda x:
                                    pd.DataFrame(rank_degenerate_values(x[id_col], x[relsize_col]),
                                    columns=[id_col, rank_col])).reset_index()

        df3 = df3[list(set(df3.columns) - {'level_3'})]
        df4 = df.merge(df3, on=[up, dn, 'ylook', id_col], how='left')

        fn = expanduser(f'{fpath}{mt}_metric_{ms}_w{w}_ranked.csv.gz')
        df4.to_csv(fn, compression='gzip')
        del df4['ylook']
        df5 = df4.drop_duplicates([up, dn, ye, pm]).sort_values([up, dn, ye, pm])
        df_agg.append(df5)

    dft = reduce(lambda df1, df2: pd.merge(df1, df2, on=[up, dn, pm, ye]), df_agg)

    if head < 0:
        dft.to_csv(expanduser(f'~/data/wos/comm_metrics/{mt}_metric_{ms}_ranked.csv.gz'),
                   compression='gzip')
    else:
        print(dft.head())
        dft.to_csv(expanduser(f'~/data/wos/comm_metrics/{mt}_metric_{ms}_ranked_tmp.csv.gz'),
                   compression='gzip')


