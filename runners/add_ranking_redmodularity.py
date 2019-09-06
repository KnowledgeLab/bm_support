from datahelpers.constants import ye, up, dn, pm
from bm_support.basic import rank_degenerate_values
from os.path import expanduser
import pandas as pd

head = 1000
# metric_sources = ['authors', 'affiliations', 'future', 'past', 'afaupa', 'afaupafu']
metric_sources = ['authors']
fpath = '~/data/wos/comm_metrics/'
windows = [1, 2, 3, None]

mt = 'redmodularity'

df_agg = []
for ms in metric_sources:
    merge_cols = [up, dn, ye, pm]
    for w in windows:
        df = pd.read_csv(expanduser(f'{fpath}{mt}_metric_{ms}_w{w}.csv.gz'))
        if head > 0:
            df = df.head(head)
        df2 = df.groupby([up, dn, 'ylook'], as_index=False, group_keys=False).apply(lambda item:
                                item.drop_duplicates('rcommid')[[up, dn, 'ylook', 'rcommrel_size', 'rcommid']])

        df3 = df2.groupby([up, dn, 'ylook']).apply(lambda x:
                                    pd.DataFrame(rank_degenerate_values(x['rcommid'], x['rcommrel_size']),
                                        columns=['rcommid', 'rank'])).reset_index()

        df3 = df3[list(set(df3.columns) - {'level_3'})]
        df4 = df.merge(df3, on=[up, dn, 'ylook', 'rcommid'], how='left')

        fn = expanduser('{0}{1}_metric_{2}_ranked.csv.gz')
        df4.to_csv(fn, compression='gzip')
        df5 = df4.set_index([up, dn, ye, pm]).sort_index()

        df_agg.append(df5)

    dft = pd.concat(df_agg, axis=1)
    print('redmod concat shape: ', dft.shape)

    if head < 0:
        dft.to_csv(expanduser(f'~/data/wos/comm_metrics/redmodularity_metric_{ms}_ranked.csv.gz'),
                   compression='gzip')
    else:
        print(dft.head())
        dft.to_csv(expanduser(f'~/data/wos/comm_metrics/redmodularity_metric_{ms}_ranked_tmp.csv.gz'),
                   compression='gzip')


