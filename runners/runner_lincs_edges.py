from os.path import expanduser, join
import pandas as pd
import numpy as np
import bm_support.cmap_tools as cte
from bm_support.cmap_tools import pt
from bm_support.gene_id_converter import GeneIdConverter, enforce_ints
from bm_support.gene_id_converter import types as gctypes


fname_sig = join(cte.data_path, cte.sig_fname)
sig_info_df = pd.read_csv(fname_sig, sep='\t', compression='gzip')

gc = GeneIdConverter(expanduser('~/data/chebi/hgnc_complete_set.json.gz'), gctypes, enforce_ints)
gc.choose_converter('entrez_id', 'symbol')
gc.update_with_broad_symbols()
gc.change_case('symbol')

gene_df_map = gc.convs[('symbol', 'entrez_id')].copy()
inv_gene_df_map = gc.convs[('entrez_id', 'symbol')].copy()

genes_gene_df = list(gc.convs[('symbol', 'entrez_id')].keys())
genes_sig_df = list(sig_info_df.pert_iname.unique())

sig_info_df.rename(columns={'pert_iname': pt}, inplace=True)

pts = list(gene_df_map.keys())
# pts = list(set(genes_gene_df) & set(sig_info_df.pt))

sig_info_df2 = sig_info_df.loc[sig_info_df[pt].isin(pts)].copy()

m1 = (sig_info_df2['pert_type'] == 'trt_oe')
m2 = (sig_info_df2['pert_itime'].apply(lambda x: float(x.split(' ')[0]) >= 6.))
m3 = (sig_info_df2['is_touchstone'] == 1)
m4 = (sig_info_df2['pert_idose'] == '1 µL') | (sig_info_df2['pert_idose'] == '2 µL') | \
    (sig_info_df2['pert_idose'] == '10 µM') | (sig_info_df2['pert_idose'] == '5 µM')
print('total number of experiments cut out by m1-m4 {0}, total number of exps {1}'.format(sum(m1 & m2 & m3 & m4),
                                                                                          sig_info_df2.shape[0]))
print('number of perts cut by m1: {0}'.format(sig_info_df2.loc[m1, pt].unique().shape[0]))
print('number of perts cut by m1 and m2: {0}'.format(sig_info_df2.loc[m1 & m2, pt].unique().shape[0]))
print('number of perts cut by m1 and m2 and m3 and m4: {0}'.format(sig_info_df2.loc[m1 & m2 & m3 & m4,
                                                                   pt].unique().shape[0]))

sig_info_df3 = sig_info_df2.loc[:].copy()
pts_working = list(sig_info_df3.pt.unique())

chunk_size = 100
chunks = [pts_working[k:k+chunk_size] for k in np.arange(0, len(pts_working), chunk_size)]

total_len = sum([len(x) for x in chunks])
print('Compare pts len vs len of chunks: ', len(pts_working), total_len)

edges_list = []
verbosity = False
# verbosity = True
df_agg = pd.DataFrame()

for chunk in chunks[:]:
    dfr = cte.get_zscore_vector(chunk, sig_info_df3, join(cte.data_path, cte.level5_fname), verbose=verbosity)
    dfr.rename(index=gene_df_map, inplace=True)
    df_agg = pd.concat([df_agg, dfr])
    # rr = cte.convert_adj_to_edges_list(dfr, verbose=verbosity)
    # edges_list.extend(rr)
    # len_proc = len(edges_list)
    len_proc = df_agg.shape[0]
    frac = 100*len_proc/len(pts_working)
    # if verbosity:
    print('Number of edges: {0}. Job: {1:.2f}% done'.format(len_proc, frac))


store = pd.HDFStore(expanduser('~/data/lincs/graph/adj_mat_all_pert_types.h5'))
store.put('df', df_agg)
