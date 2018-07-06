import gzip
import json
from itertools import permutations
import pandas as pd
from cmapPy.pandasGEXpress.parse import parse
from os.path import expanduser, join
from numpy import argmax, ogrid, abs
from scipy.stats import norm


# types = ['hgnc_id', 'entrez_id', 'symbol']
# enforce_ints = ['entrez_id']

# pt stands for perturbagen
pt = 'pt'

data_path = expanduser('~/data/lincs')

level5_fname = 'GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n476251x12328.gctx'
sig_fname = 'GSE92742_Broad_LINCS_sig_info.txt.gz'

pairs = [(4193, 7157), (4792, 4790), (4790, 7124)]


def hack_binarize_list(str_list):
    return ["b'{0}'".format(x) for x in str_list]


def hack_unbinarize_list(str_list, to_int=False):
    if to_int:
        return [int(x[2:-1]) for x in str_list]
    else:
        return [x[2:-1] for x in str_list]


def get_zscore_vector(chunk, sig_df, gctx_fname,
                      mean_agg_columns=(pt, 'pert_idose'),
                      verbose=False):
    # get binarized perts
    # bin_chunk = hack_binarize_list(chunk)

    # get all sig_ids for these perts
    mask = sig_df.pt.isin(chunk)

    if verbose:
        print('For {0} perturbagens : number of experiments is {1}'.format(len(chunk), sum(mask)))
    bin_sigs = hack_binarize_list(sig_df.loc[mask, 'sig_id'])
    level5_gctoo = parse(gctx_fname, cid=bin_sigs)
    dfr = level5_gctoo.data_df

    if verbose:
        print('Shape[0] of level5 df is {0}'.format(dfr.shape[0]))
    dfr = dfr.rename(columns=dict(zip(dfr.columns, hack_unbinarize_list(dfr.columns))),
                     index=dict(zip(dfr.index, hack_unbinarize_list(dfr.index, True))))
###
    dfw = pd.merge(dfr.T, sig_df, left_index=True, right_on='sig_id', how='inner')
###
    # take mean for a given pertubagen, pert_type, time and dose (across cellline) by default
    if all([(x in dfw.columns) for x in mean_agg_columns]):
        expression_cols = list(set(dfw.columns) - {'sig_id', 'pert_id', 'is_touchstone'})
        dfw2 = dfw[expression_cols].groupby(mean_agg_columns).apply(lambda x: x.mean())
    else:
        raise ValueError('mean_agg_columns are incompatible with df')

    # groupby by first level index (should be pt) and take max
    dfw3 = dfw2.groupby(level=0).apply(lambda x:
                                       pd.Series(x.values[argmax(abs(x.values), axis=0),
                                                          ogrid[:x.shape[1]]]))
    if verbose:
        print('Shape[1] of output df is {0} [should be the same as level5 df]'.format(dfw3.shape[1]))

    return dfw3


def convert_adj_to_edges_list(adj, thr_fraction=None, weighted=True, verbose=True):
    # weighted False and thr_fraction None makes no sense
    # thr_fraction is the percentile below which to cut off
    edges_df = adj.stack()
    if thr_fraction:
        thr = norm.ppf(thr_fraction)
        mask = (edges_df.abs() > thr)
        if verbose:
            print('Number of edges above threshold from {1}: {0}'.format(sum(mask), mask.shape[0]))
        edges_df = edges_df.loc[mask]
    edges = edges_df.reset_index()
    if verbose:
        print('edges dtypes {0}'.format(edges.dtypes))
    ups, dns, vals = edges[edges.columns[0]], edges[edges.columns[1]], edges[edges.columns[2]]
    if weighted:
        ret_list = list(zip(ups, dns, vals))
    else:
        ret_list = list(zip(ups, dns))
    return ret_list


def extract_expressions(converter, sig_info, gene_info, up_down_pairs, level5_fname):
    #TODO fix : should go entrez_id -> gene_id(gene_info)

    ups = list(set(map(lambda x: x[0], up_down_pairs)))
    downs = list(set(map(lambda x: x[1], up_down_pairs)))
    converter.choose_converter('entrez_id', 'symbol')
    # what if converter does not have some ups?
    ss = set(converter.convs['entrez_id', 'symbol'].keys())
    # non convertibles ups
    ups_out = set(ups) - ss
    ups_rem = list(set(ups) & ss)
    names = [converter[i] for i in ups_rem]

    names_out = set(names) - set(sig_info['pert_iname'].unique())
    names_rem = list(set(names) & set(sig_info['pert_iname'].unique()))
    # ups not in the experiment
    converter.choose_converter('symbol', 'entrez_id')
    ups_out2 = [converter[i] for i in names_out]
    ups_out3 = list(set(ups_out2) | set(ups_out))
    # ups are translated into sigs
    mask = sig_info['pert_iname'].isin(names_rem)
    si = sig_info[mask].copy()
    converter.choose_converter('symbol', 'entrez_id')
    si['pert_eid'] = si['pert_iname'].apply(lambda x: converter[x])
    si_ids = list(si['sig_id'])

    set_genes_ids = set(gene_info['pr_gene_id'].unique())
    downs_out = set(downs) - set_genes_ids
    downs_rem = list(set(downs) & set_genes_ids)

    si_ids2 = ["b'{0}'".format(x) for x in si_ids]
    downs_rem2 = ["b'{0}'".format(x) for x in downs_rem]

    level5_gctoo = parse(level5_fname, cid=si_ids2, rid=downs_rem2)

    df = level5_gctoo.data_df.unstack().reset_index().rename(columns={'cid': 'sig_id', 0: 'score'})
    dfr = pd.merge(si, df, on=['sig_id'], how='right')
    dfr.rename(columns={'pert_eid': 'up', 'rid': 'dn'}, inplace=True)

    out_pairs = list(filter(lambda x: x[0] in ups_out3 or x[1] in downs_out, up_down_pairs))

    pp = pd.DataFrame(up_down_pairs, columns=['up', 'dn'])
    dff = pd.merge(dfr, pp, on=['up', 'dn'])
    return dff, out_pairs


# class GeneIdConverter(object):
#
#     def __init__(self, fpath, types_list, enforce_int):
#         """
#
#         :param fpath: file path to gzipped json
#             json format is like the one taken from
#                 ftp://ftp.ebi.ac.uk/pub/databases/genenames/new/json/hgnc_complete_set.json
#         :param types_list: can be from
#             ['hgnc_id', 'entrez_id', 'symbol', 'cosmic', 'ucsc_id']
#
#         :param enforce_int: list of types for which type int is enforced, e.g. 'entrez_id'
#         """
#
#         with gzip.open(fpath, 'rb') as f:
#             jsonic = json.loads(f.read().decode('utf-8'))
#         self.convs = {}
#         self.sets = {}
#         types_perms = list(permutations(types_list, 2))
#
#         for u in types_list:
#             s = filter(lambda x: u in x.keys(), jsonic['response']['docs'])
#             self.sets[u] = set(map(lambda x: int(x[u]) if u in enforce_int else x[u], s))
#
#         for u, v in types_perms:
#             ff = filter(lambda x: u in x.keys() and v in x.keys(), jsonic['response']['docs'])
#             self.convs[u, v] = dict(map(lambda x: (int(x[u]) if u in enforce_int else x[u],
#                                                    int(x[v]) if v in enforce_int else x[v]), ff))
#
#         self.types_list = types_list
#         self.types_perms = types_perms
#         self.enforce_int = enforce_int
#
#         self.u, self.v = types_list[0], types_list[1]
#
#     def choose_converter(self, u, v):
#         if u in self.types_list and v in self.types_list:
#             self.u, self.v = u, v
#
#     def __getitem__(self, key):
#         return self.convs[(self.u, self.v)][key]
#

# gc = GeneIdConverter('/Users/belikov/data/chebi/hgnc_complete_set.json.gz', types, enforce_ints)
# gc.choose_converter('entrez_id', 'symbol')


