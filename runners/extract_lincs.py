import pandas as pd
import argparse
import numpy as np
import time
import bm_support.cmap_tools as cte
from bm_support.cmap_tools import pt
from cmapPy.pandasGEXpress.parse import parse
from datahelpers.constants import up, dn
from bm_support.gene_id_converter import GeneIdConverter, enforce_ints
from bm_support.gene_id_converter import types as gctypes

from os.path import join
from os.path import expanduser

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--datasource',
                    default='gw',
                    help='type of data to work with [gw, lit]')

parser.add_argument('-v', '--version',
                    default=8, type=int,
                    help='version of data source')

parser.add_argument('-s', '--batchsize',
                    default=1000, type=int,
                    help='size of data batches')

parser.add_argument('-n', '--maxsize-sequence',
                    default=20, type=int,
                    help='version of data source')

parser.add_argument('-p', '--partition-sequence',
                    nargs='+', default=[0.1, 0.9], type=float,
                    help='define interval of observed freqs for sequence consideration')

parser.add_argument('-t', '--test',
                    default='-1', type=int,
                    help='test mode: number of pairs to look at')

parser.add_argument('-c', '--chunk-size',
                    default=200, type=int,
                    help='chunk size of genes query')

parser.add_argument('-u', '--subversion',
                    default=3, type=int,
                    help='subversion of protein id maps')

args = parser.parse_args()
print(args)
origin = args.datasource
a, b = args.partition_sequence
n = args.maxsize_sequence
version = args.version
chunk_size = args.chunk_size
test_size = args.test
subversion = args.subversion
verbosity = True

sig_info_df = pd.read_csv(join(cte.data_path, cte.sig_fname), sep='\t', compression='gzip')
sig_info_df.rename(columns={'pert_iname': pt}, inplace=True)

df_pairs = pd.read_csv(expanduser('~/data/kl/claims/pairs_freq_{0}_'
                                  'v_{1}_n_{2}_a_{3}_b_{4}.csv.gz'.format(origin, version, n, a, b)),
                       compression='gzip', index_col=0)

fname_gene = '~/data/lincs/GSE92742_Broad_LINCS_gene_info.txt.gz'
gene_df = pd.read_csv(expanduser(fname_gene), sep='\t')
gene_df.rename(columns={'pr_gene_symbol': pt}, inplace=True)

# protein (name) : entrez_id
gc = GeneIdConverter(expanduser('~/data/chebi/hgnc_complete_set.json.gz'), gctypes, enforce_ints)
gc.choose_converter('entrez_id', 'symbol')
gc.update_with_broad_symbols()
gc.change_case('symbol')

gene_df_map = gc.convs[('symbol', 'entrez_id')].copy()
inv_gene_df_map = gc.convs[('entrez_id', 'symbol')].copy()

pairs = [(x[0], x[1]) for x in df_pairs[['up', 'dn']].values]
pairs_df = pd.DataFrame(pairs, columns=[up, dn])

ups_set = set([x[0] for x in pairs])
pids_working = list(set(inv_gene_df_map.keys()) & ups_set)

ups = list(set([x[0] for x in pairs]))
pts = [inv_gene_df_map[x] for x in ups if x in inv_gene_df_map.keys()]

if verbosity:
    print('len pts: {0}; unique: {1} '.format(len(pts), len(set(pts))))
    print(sig_info_df.loc[sig_info_df.pt.isin(pts), pt].unique().shape)


sig_info_df2 = sig_info_df.loc[sig_info_df.pt.isin(pts)].copy()
sig_df = sig_info_df2.loc[:].copy()
sig_df_map = dict(sig_df[['sig_id', pt]].values)

pts_working = list(sig_info_df2[pt].unique())
pts_working2 = list(set([gc[x] for x in pids_working]) & set(pts_working))

pts = pts_working2

if not pts:
    raise ValueError('the intersection of databases is trivial!')

if test_size > 0:
    pts = pts[:test_size]

chunks = [pts[k:k+chunk_size] for k in np.arange(0, len(pts), chunk_size)]
total_len = sum([len(x) for x in chunks])
print('Compare pts len vs len of chunks: ', len(pts), total_len)

time_prev = time.time()
gene_df_map_df = pd.DataFrame(list(gene_df_map.items()), columns=[pt, up])

dfs = []

df_agg = pd.DataFrame()

if verbosity:
    print('chn size: {0}'.format(chunk_size))

ups_procd = 0

for chunk in chunks[:2]:
    mask = sig_df[pt].isin(chunk)
    if verbosity:
        print('partial view of chunk: {0}'.format(chunk[:5]))
    if verbosity:
        print('sum sig mask: {0}'.format(sum(mask)))

    bin_sigs = cte.hack_binarize_list(sig_df.loc[mask, 'sig_id'])
    level5_gctoo = parse(join(cte.data_path, cte.level5_fname), cid=bin_sigs)

    dfr = level5_gctoo.data_df
    dfr = dfr.rename(columns=dict(zip(dfr.columns, cte.hack_unbinarize_list(dfr.columns))),
                     index=dict(zip(dfr.index, cte.hack_unbinarize_list(dfr.index, True))))

    dfr2 = dfr.unstack().reset_index().rename(columns={'cid': 'sig_id', 0: 'score', 'rid': dn})

    dfw = pd.merge(dfr2, sig_df, on='sig_id', how='inner')
    del dfr2
    dfw2 = pd.merge(dfw, gene_df_map_df, on=pt, how='inner')
    dfr3 = pd.merge(dfw2, pairs_df, how='inner', on=[up, dn])

    set_cols = set(dfr3.columns) - {up, dn, 'score'}
    ordered_cols = list(set_cols) + [up, dn, 'score']
    time_cur = time.time()
    delta_time_sec = time_cur - time_prev

    time_prev = time_cur
    df_agg = pd.concat([df_agg, dfr3[ordered_cols]])
    ups_procd += len(chunk)
    frac = 100*ups_procd/len(pts)
    if verbosity:
        print('Job: {0:.2f}% done. {1} {2}'.format(frac, ups_procd, len(df_agg[up].unique())))
        print('Iteration took {0:.1f} sec'.format(delta_time_sec))

# df_agg.to_csv(expanduser('~/data/kl/claims/'
#                          'lincs_{0}_v_{1}_n_{2}_'
#                          'a_{3}_b_{4}_sv_{5}.csv.gz'.format(origin, version, n, a, b, subversion)),
#               compression='gzip', index=False)
