from os.path import expanduser, join
import pandas as pd
import numpy as np
import bm_support.cmap_tools as cte
from bm_support.cmap_tools import pt

import gzip
import pickle
# from cmapPy.pandasGEXpress.parse import parse


fname_sig = join(cte.data_path, cte.sig_fname)
sig_info_df = pd.read_csv(fname_sig, sep='\t', compression='gzip')

fname_gene = '~/data/lincs/GSE92742_Broad_LINCS_gene_info.txt.gz'
gene_df = pd.read_csv(expanduser(fname_gene), sep='\t')

genes_gene_df = list(gene_df.pr_gene_id.unique())
genes_sig_df = list(sig_info_df.pert_iname.unique())

# pt is get_title in gene_df or pert_iname
gene_df.rename(columns={'pr_gene_symbol': pt}, inplace=True)
sig_info_df.rename(columns={'pert_iname': pt}, inplace=True)
gene_df[pt] = gene_df[pt].apply(lambda x: x.lower())
sig_info_df[pt] = sig_info_df[pt].apply(lambda x: x.lower())

gene_df_map = dict(gene_df[['pt', 'pr_gene_id']].values)

pts = list(set(gene_df.pt) & set(sig_info_df.pt))

sig_info_df2 = sig_info_df.loc[sig_info_df.pt.isin(pts)].copy()
gene_df2 = gene_df.loc[gene_df.pt.isin(pts)].copy()

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
    len_proc = len(edges_list)
    len_proc = df_agg.shape[0]
    frac = 100*len_proc/len(pts_working)
    if verbosity:
        print('Number of edges: {0}. Job: {1:.2f}% done'.format(len_proc, frac))

# with gzip.open(expanduser('~/data/lincs/graph/edges_all.pgz'), 'wb') as fp:
#     pickle.dump(edges_list, fp)

store = pd.HDFStore(expanduser('~/data/lincs/graph/adj_mat_all_pert_types.h5'))
store.put('df', df_agg)
