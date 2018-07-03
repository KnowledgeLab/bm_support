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

sig_info_df = pd.read_csv(join(cte.data_path, cte.sig_fname), sep='\t', compression='gzip')
sig_info_df.rename(columns={'pert_iname': pt}, inplace=True)

df_pairs = pd.read_csv(expanduser('~/data/kl/claims/pairs_freq_{0}_'
                                  'v_{1}_n_{2}_a_{3}_b_{4}.csv.gz'.format(origin, version, n, a, b)),
                       compression='gzip', index_col=0)

fname_gene = '~/data/lincs/GSE92742_Broad_LINCS_gene_info.txt.gz'
gene_df = pd.read_csv(expanduser(fname_gene), sep='\t')
gene_df.rename(columns={'pr_gene_symbol': pt}, inplace=True)

# protein (name) : entrez_id
# gene_df_map_ = dict(gene_df[['pt', 'pr_gene_id']].values)
# inv_gene_df_map_ = dict(gene_df[['pr_gene_id', 'pt']].values)
gc = GeneIdConverter(expanduser('~/data/chebi/hgnc_complete_set.json.gz'), gctypes, enforce_ints)
gc.choose_converter('entrez_id', 'symbol')
gc.update_with_broad_symbols()
gc.change_case('symbol')

gene_df_map = gc.convs[('symbol', 'entrez_id')].copy()
inv_gene_df_map = gc.convs[('entrez_id', 'symbol')].copy()

# print(list(gene_df_map.items())[:5])
# print(list(gene_df_map_.items())[:5])
#
# gene_df_map = gene_df_map_
# inv_gene_df_map = inv_gene_df_map_

pairs = [(x[0], x[1]) for x in df_pairs[['up', 'dn']].values]
pairs_df = pd.DataFrame(pairs, columns=[up, dn])

ups = list(set([x[0] for x in pairs]))
pts = [inv_gene_df_map[x] for x in ups if x in inv_gene_df_map.keys()]

if not pts:
    raise ValueError('the intersection of databases is trivial!')

if test_size > 0:
    pts = pts[:test_size]

chunks = [pts[k:k+chunk_size] for k in np.arange(0, len(pts), chunk_size)]

sig_info_df2 = sig_info_df.loc[sig_info_df.pt.isin(pts)].copy()
sig_df = sig_info_df2.loc[:].copy()
sig_df_map = dict(sig_df[['sig_id', pt]].values)

time_prev = time.time()

dfs = []

verbosity = True
df_agg = pd.DataFrame()

if verbosity:
    print('len pts: {0}'.format(len(pts)))
    print('chn size: {0}'.format(chunk_size))

ups_procd = 0

for chunk in chunks[:]:
    mask = sig_df.pt.isin(chunk)
    if verbosity:
        print('partial view of chunk: {0}'.format(chunk[:5]))
    if verbosity:
        print('sum sig mask: {0}'.format(sum(mask)))

    bin_sigs = cte.hack_binarize_list(sig_df.loc[mask, 'sig_id'])
    level5_gctoo = parse(join(cte.data_path, cte.level5_fname), cid=bin_sigs)

    dfr = level5_gctoo.data_df

    # if verbosity:
    #     print('dfr size: {0}'.format(dfr.shape))

    dfr = dfr.rename(columns=dict(zip(dfr.columns, cte.hack_unbinarize_list(dfr.columns))),
                     index=dict(zip(dfr.index, cte.hack_unbinarize_list(dfr.index, True))))

    dfr2 = dfr.unstack().reset_index().rename(columns={'cid': 'sig_id', 0: 'score'})
    dfr2[up] = dfr2['sig_id'].apply(lambda x: gene_df_map[sig_df_map[x]])
    dfr2 = dfr2.rename(columns={'rid': dn})
    dfr2_cut = pd.merge(dfr2, pairs_df, how='inner', on=[up, dn])
    dfr3 = dfr2_cut.merge(sig_df, on='sig_id', how='left')
    set_cols = set(dfr3.columns) - {up, dn, 'score'}
    time_cur = time.time()
    delta_time_sec = time_cur - time_prev

    time_prev = time_cur
    df_agg = pd.concat([df_agg, dfr3[list(set_cols) + [up, dn, 'score']]])
    ups_procd += len(chunk)
    frac = 100*ups_procd/len(pts)
    if verbosity:
        print('Job: {0:.2f}% done. {1} {2}'.format(frac, ups_procd, len(df_agg[up].unique())))
        print('Iteration took {0:.1f} sec'.format(delta_time_sec))

# if test_size < 0:
df_agg.to_csv(expanduser('~/data/kl/claims/'
                         'lincs_{0}_v_{1}_n_{2}_'
                         'a_{3}_b_{4}_sv_{5}.csv.gz'.format(origin, version, n, a, b, subversion)),
              compression='gzip', index=False)
