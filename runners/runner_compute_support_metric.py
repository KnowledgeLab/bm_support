import pandas as pd
from datahelpers.constants import ye, up, dn, pm, aus
from bm_support.bigraph_support import compute_support_index, compute_affinity_index, compute_modularity_index
from bm_support.add_features import get_mapping_data
from os.path import expanduser
import time
import argparse
import random


def main(flag_dict, head, metric_type, verbosity=False):
    random.seed(13)
    fraction_imporant_v_vertices = 0.5
    window_sizes = [None, 1, 2, 3]
    window_sizes = [None]
    disjoint_uv = False

    df_pm_wid = pd.read_csv(expanduser('~/data/wos/cites/wosids.csv.gz'), index_col=0)
    dfy = pd.read_csv(expanduser('~/data/wos/pmids/updnyearpmid_all.csv.gz'), index_col=0)

    if verbosity:
        print('metric type: {0}'.format(metric_type))
    uv_dict, pm_wid_dict = get_mapping_data(metric_type, dfy, df_pm_wid)
    if metric_type == 'affiliations' or metric_type == 'authors':
        use_wosids = False
        fraction_imporant_v_vertices = 0.5
    elif metric_type == 'past' or metric_type == 'future':
        use_wosids = True
        fraction_imporant_v_vertices = 0.2
    else:
        use_wosids = True

    if verbosity:
        print('number of rows in dfy: {0}'.format(dfy.shape[0]))
        print('len pm_wid_dict {0}'.format(len(pm_wid_dict)))
        print('len pm_aus_dict {0}'.format(len(uv_dict)))

    if head > 0:
        dfy = dfy.head(head)

    df_agg_supp = []
    df_agg_aff = []
    df_agg_mod = []
    df_agg_redmod = []

    times = [time.time()]

    for window_size in window_sizes:
        if flag_dict['support']:
            dfr = dfy.groupby([up, dn]).apply(lambda x: compute_support_index(x, uv_dict,
                                                                              pm_wid_dict, ye, window_size,
                                                                              fraction_imporant_v_vertices,
                                                                              use_wosids=use_wosids))
            dfr = dfr.reset_index()

            dfr = dfr.set_index([up, dn, ye])
            print(dfr.head())
            df_agg_supp.append(dfr)
            times.append(time.time())
            print('supp, window size {0} {1:.2f} sec elapsed'.format(window_size, times[-1] - times[-2]))

        if flag_dict['affinity']:
            dfr2 = dfy.groupby([up, dn]).apply(lambda x: compute_affinity_index(x, uv_dict, pm_wid_dict,
                                                                                ye, window_size,
                                                                                use_wosids=use_wosids))

            dfr2 = dfr2.reset_index()

            dfr2 = dfr2.drop(['level_2'], axis=1)

            dfr2[pm] = dfr2[pm].astype(int)
            dfr2[ye] = dfr2[ye].astype(int)
            dfr2 = dfr2.set_index([up, dn, ye, pm]).sort_index()
            print(dfr2.head())
            df_agg_aff.append(dfr2)
            times.append(time.time())
            print('aff, window size {0} {1:.2f} sec elapsed'.format(window_size, times[-1] - times[-2]))

        if flag_dict['mod']:
            random.seed(13)

            print('***')
            print(len(uv_dict), len(pm_wid_dict), ye, window_size, use_wosids, disjoint_uv, dfy.shape)
            sorted_keys = sorted(uv_dict.keys())
            print([(k, (len(uv_dict[k]), sorted(uv_dict[k])[:5])) for k in sorted_keys[-5:]])
            dfr3 = dfy.groupby([up, dn]).apply(lambda x: compute_modularity_index(x, uv_dict, pm_wid_dict,
                                                                                  ye, window_size,
                                                                                  use_wosids=use_wosids,
                                                                                  disjoint_uv=disjoint_uv,
                                                                                  verbose=verbosity))

            print(dfr3.shape)
            dfr3 = dfr3.reset_index()

            dfr3 = dfr3.drop(['level_2'], axis=1)

            dfr3[pm] = dfr3[pm].astype(int)
            dfr3[ye] = dfr3[ye].astype(int)
            dfr3 = dfr3.set_index([up, dn, ye, pm]).sort_index()
            print(dfr3.head())
            df_agg_mod.append(dfr3)
            times.append(time.time())
            print('mod, window size {0} {1:.2f} sec elapsed'.format(window_size, times[-1] - times[-2]))

        # modularity reduced
        if flag_dict['redmod']:
            random.seed(13)

            dfr4 = dfy.groupby([up, dn]).apply(lambda x: compute_modularity_index(x, uv_dict, pm_wid_dict,
                                                                                  ye, window_size,
                                                                                  use_wosids=use_wosids,
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

    if flag_dict['support']:
        dft = pd.concat(df_agg_supp, axis=1)
        print('supp concat shape: ', dft.shape)
        print('Support: fractions of indices that are non zero:')
        for c in [col for col in dft if 'ind' in col]:
            print('{0} : {1:.2f} %'.format(c, 100*sum(dft[c] != 0)/dft.shape[0]))
        if head < 0:
            dft.to_csv(expanduser('~/data/wos/cites/support_metric_{0}.csv.gz'.format(metric_type)),
                       compression='gzip')
        else:
            print(dft.head())
            dft.to_csv(expanduser('~/data/wos/cites/support_metric_{0}_tmp2.csv.gz'.format(metric_type)),
                       compression='gzip')

    if flag_dict['affinity']:
        dft = pd.concat(df_agg_aff, axis=1)
        print('aff concat shape: ', dft.shape)
        print('Affinity: fractions of indices that are non zero:')
        for c in dft.columns:
            print('{0} : {1:.2f} %'.format(c, 100 * sum(dft[c] != 0) / dft.shape[0]))
        if head < 0:
            dft.to_csv(expanduser('~/data/wos/cites/affinity_metric_{0}.csv.gz'.format(metric_type)),
                       compression='gzip')
        else:
            print(dft.head())
            dft.to_csv(expanduser('~/data/wos/cites/affinity_metric_{0}_tmp2.csv.gz'.format(metric_type)),
                       compression='gzip')

    if flag_dict['mod']:
        dft = pd.concat(df_agg_mod, axis=1)
        print('mod concat shape: ', dft.shape)
        print('Modularity: fractions of indices that are non one:')
        for c in dft.columns:
            print('{0} : {1:.2f} %'.format(c, 100 * sum(dft[c] != 1) / dft.shape[0]))
        if head < 0:
            dft.to_csv(expanduser('~/data/wos/cites/modularity_metric_{0}.csv.gz'.format(metric_type)),
                       compression='gzip')
        else:
            print(dft.head())
            dft.to_csv(expanduser('~/data/wos/cites/modularity_metric_{0}_tmp2.csv.gz'.format(metric_type)),
                       compression='gzip')

    if flag_dict['redmod']:
        dft = pd.concat(df_agg_redmod, axis=1)
        print('mod concat shape: ', dft.shape)
        print('Modularity: fractions of indices that are non one:')
        for c in dft.columns:
            print('{0} : {1:.2f} %'.format(c, 100 * sum(dft[c] != 1) / dft.shape[0]))
        if head < 0:
            dft.to_csv(expanduser('~/data/wos/cites/redmodularity_metric_{0}.csv.gz'.format(metric_type)),
                       compression='gzip')
        else:
            print(dft.head())
            dft.to_csv(expanduser('~/data/wos/cites/redmodularity_metric_{0}_tmp2.csv.gz'.format(metric_type)),
                       compression='gzip')


if __name__ == "__main__":

    """
    metric_type can be 'affiliations', authors', 'past', 'future'
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--metric-type',
                        default='affiliations')

    parser.add_argument('-m', '--metrics',
                        nargs='*',
                        default=['support', 'affinity', 'mod', 'redmod'])

    parser.add_argument('--head',
                        default='2000', type=int,
                        help='take head rows')

    parser.add_argument('-v', '--verbosity',
                        default=False,
                        help='verbosity, True or False')

    args = parser.parse_args()
    metrics_dict = {k: True for k in args.metrics}
    extra = {k: False for k in list({'support', 'affinity', 'mod', 'redmod'} - set(args.metrics))}
    metrics_dict.update(extra)
    main(metrics_dict, args.head, args.metric_type, args.verbosity)
