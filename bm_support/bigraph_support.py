import pandas as pd
import numpy as np
from itertools import combinations
from datahelpers.constants import iden, ye, ai, ps, up, dn, ar, ni, cexp, qcexp, dist, rdist, pm
import igraph as ig
import random


def compute_support_index(data, bipart_edges_dict, pm_wid_dict, window_col=ye,
                          window=None, frac_important=0.1,
                          transform='square', mode='all', use_wosids=True,
                          verbose=False):
    """

    :param data:
            DataFrame of format:
                columns=[pm, window_col]

    :param bipart_edges_dict:
            {id_n: [id_k]}
    :param pm_wid_dict:
            {pm: id}
    :param window_col:
            column on which to window // partition into groups
    :param window:
            numerical value of the window
    :param frac_important:
            fraction of the important papers
    :param transform: 'linear' or 'square', applied to degree of vertex set V
    :param mode: 'cross' for support index for the intersection of U and V
    :param use_wosids:
    :param verbose:
    :return:
    """

    ixs = sorted(data[window_col].unique())
    r_agg = []
    ind_agg = []
    for ix in ixs:
        if window:
            mask = (data[window_col] <= ix) & (data[window_col] > ix - window)
        else:
            mask = (data[window_col] <= ix)
        cur_pmids = data.loc[mask, pm].unique()
        if use_wosids:
            cur_wosids = [pm_wid_dict[k] for k in cur_pmids if k in pm_wid_dict.keys()]
        else:
            cur_wosids = list(set(cur_pmids))
        if mode == 'all':
            cur_cites = [list(set(bipart_edges_dict[k])) for k in cur_wosids]
        elif mode == 'cross':
            cur_cites = [[y for y in list(set(bipart_edges_dict[k])) if y in cur_wosids] for k in cur_wosids]

        if verbose:
            print(ix, cur_cites)

        alpha, n_unique_citations, n_edges = compute_support(cur_cites, frac_important, transform)
        ind_agg.append(ix)
        r_agg.append((alpha, len(cur_cites), n_unique_citations, n_edges))

    if window:
        suff = '{0}'.format(window)
    else:
        suff = ''
    cols = ['suppind', 'size_level_a', 'size_level_b', 'n_edges']
    ren_cols = ['{0}{1}'.format(k, suff) for k in cols]
    dfr = pd.DataFrame(r_agg, index=ind_agg, columns=ren_cols)
    dfr = dfr.rename_axis(window_col, axis='index')
    return dfr


# TODO this function has to be double checked, the filters might behave unpredictably

def compute_affinity_index(data, bipart_edges_dict, pm_wid_dict, window_col=ye,
                           window=None, mode='all', use_wosids=True, verbose=False):
    """
    :param data:
            DataFrame of format:
                columns=[pm, window_col]

    :param bipart_edges_dict:
            {id_n: [id_k]}
    :param pm_wid_dict:
            {pm: id}
    :param window_col:
            column on which to window // partition into groups
    :param window:
            numerical value of the window
    :param mode: 'cross' for support index for the intersection of U and V
    :return:
    """

    ixs = sorted(data[window_col].unique())
    r_agg = []
    for ix in ixs:
        mask = (data[window_col] <= ix)
        if window:
            mask &= (data[window_col] > ix - window)

        pmids = list(data.loc[mask, pm].unique())
        pmids_ix = list(data.loc[data[window_col] == ix, pm].unique())
        if use_wosids:
            pmids_present = [pmx for pmx in pmids if pmx in pm_wid_dict.keys()]
            pmids_ix = [pmx for pmx in pmids_ix if pmx in pmids_present]
            wosids_present = [pm_wid_dict[k] for k in pmids_present]
        else:
            pmids_present = pmids
            wosids_present = pmids
        if pmids_ix:
            if mode == 'cross':
                cur_cites = [[y for y in list(set(bipart_edges_dict[k])) if y in wosids_present] for k in wosids_present]
            else:
                cur_cites = [list(set(bipart_edges_dict[k])) for k in wosids_present]
            alphas = compute_affinity(cur_cites)
            alphas_dict = dict(zip(pmids_present, alphas))
            output = [(ix, j, alphas_dict[j]) for j in pmids_ix]

            if output:
                r_agg.append(output)

    if window:
        col_suffixed = 'affind{0}'.format(window)
    else:
        col_suffixed = 'affind'

    if r_agg:
        rdata = np.concatenate(r_agg)
        dfr = pd.DataFrame(rdata, columns=[window_col, pm, col_suffixed])
    else:
        dfr = pd.DataFrame(columns=[window_col, pm, col_suffixed])
    return dfr


def compute_modularity_index(data, bipart_edges_dict, pm_wid_dict, window_col=ye, window=None, mode='all',
                             disjoint_uv=True, modularity_mode='uv', verbose=False):
    """

    :param data:
            DataFrame of format:
                columns=[pm, window_col]

    :param bipart_edges_dict:
            {id_n: [id_k]}
    :param pm_wid_dict:
            {pm: id}
    :param window_col:
            column on which to window // partition into groups
    :param window:
            numerical value of the window
    :param mode: 'cross' for support index for the intersection of U and V
    :param disjoint_uv: treat U and V  as disjoint sets
    :param modularity_mode: uv for u-v bipartite graph communities, u for U graph communities (weight from uv structure)
    :param verbose: verbosity level
    :return:
    """

    ixs = sorted(data[window_col].unique())
    r_agg = []
    for ix in ixs:
        if window:
            mask = (data[window_col] <= ix) & (data[window_col] > ix - window)
        else:
            mask = (data[window_col] <= ix)

        pmids = list(data.loc[mask, pm].unique())
        pmids_ix = list(data.loc[data[window_col] == ix, pm].unique())
        if verbose:
            print('***')
            print('up, dn, ye: {0} {1}'.format(list(data[[up, dn]].iloc[0].values), ix))

        if pm_wid_dict:
            pmids_present = [pmx for pmx in pmids if pmx in pm_wid_dict.keys()]
            pmids_ix = [pmx for pmx in pmids_ix if pmx in pmids_present]
            wosids_present = [pm_wid_dict[k] for k in pmids_present]
        else:
            pmids_present = pmids
            wosids_present = pmids
        if pmids_ix:
            if mode == 'cross':
                uvs_list = [(k, [y for y in list(set(bipart_edges_dict[k])) if y in wosids_present])
                            for k in wosids_present]
            else:
                uvs_list = [(k, list(set(bipart_edges_dict[k]))) for k in wosids_present]

            if modularity_mode == 'uv':
                commsize_ncomm_usize = compute_comm_structure_bigraph(uvs_list, disjoint_uv, verbose)
                csize_dict = dict(zip(pmids_present, commsize_ncomm_usize))
                output = [(ix, p, *csize_dict[p]) for p in pmids_ix]
            else:
                commsize_ncomm_usize = compute_comm_structure_reduced_graph(uvs_list, verbose)
                csize_dict = dict(zip(pmids_present, commsize_ncomm_usize))
                output = [(ix, p, *csize_dict[p]) for p in pmids_present]
            # ***
            if output:
                r_agg.append(output)

    if window:
        suff = '{0}'.format(window)
    else:
        suff = ''

    if modularity_mode == 'uv':
        cols = ['comm_size', 'ncomms', 'size_ulist', 'ncomponents', 'commproj_size', 'commprojrel_size']
        cols = ['{0}{1}'.format(k, suff) for k in cols]
        # cols = []
    else:
        cols = ['rcomm_size', 'rncomms', 'rncomponents', 'rcommid', 'rcommrel_size']

    if r_agg:
        rdata = np.concatenate(r_agg)
        dfr = pd.DataFrame(rdata, columns=[window_col, pm, *cols])
    else:
        dfr = pd.DataFrame(columns=[window_col, pm, *cols])
    return dfr


def compute_modularity_index_gen(data, bipart_edges_dict, window_col=ye, window=None, mode='all', disjoint_uv=True,
                                 modularity_mode='uv', verbose=False):
    """

    :param data:
            DataFrame of format:
                columns=[pm, window_col]

    :param bipart_edges_dict:
            {id_n: [id_k]}
    :param window_col:
            column on which to window // partition into groups
    :param window:
            numerical value of the window
    :param mode: 'cross' for support index for the intersection of U and V
    :param disjoint_uv: treat U and V  as disjoint sets
    :param modularity_mode: uv for u-v bipartite graph communities, u for U graph communities (weight from uv structure)
    :param verbose: verbosity level
    :return:
    """

    ixs = sorted(data[window_col].unique())
    r_agg = []
    if verbose:
        print('***')
        print('data shape: {0}, columns {1}'.format(data.shape, data.columns))
        print('ixs: {0}'.format(ixs))
        print('up, dn: {0}'.format(data.drop_duplicates([up, dn])[[up, dn]]))

    for ii in range(len(ixs)):
        if window:
            lower = max(0, ii - window + 1)
            mask = data[window_col].isin(ixs[lower:(ii + 1)])
        else:
            mask = data[window_col].isin(ixs[:(ii + 1)])
        ix = ixs[ii]

        pmids = list(data.loc[mask, pm].unique())
        pmids_ix = list(data.loc[data[window_col] == ix, pm].unique())
        # if verbose:
        #     print('***')
        #     print('up, dn, ye: {0} {1}'.format(list(data[[up, dn]].iloc[0].values), ix))
        if pmids_ix:
            if mode == 'cross':
                uvs_list = [(k, [y for y in list(set(bipart_edges_dict[k])) if y in pmids])
                            for k in pmids]
            else:
                uvs_list = [(k, list(set(bipart_edges_dict[k])))  if k in bipart_edges_dict.keys()
                            else (k, []) for k in pmids]
            if modularity_mode == 'uv':
                commsize_ncomm_usize = compute_comm_structure_bigraph(uvs_list, disjoint_uv, verbose)
                csize_dict = dict(zip(pmids, commsize_ncomm_usize))
                output = [(ix, p, *csize_dict[p]) for p in pmids_ix]
            else:
                commsize_ncomm_usize = compute_comm_structure_reduced_graph(uvs_list, verbose)
                csize_dict = dict(zip(pmids, commsize_ncomm_usize))
                output = [(ix, p, *csize_dict[p]) for p in pmids]
            # ***
            if output:
                r_agg.append(output)

    if window:
        suff = f'{window}'
    else:
        suff = ''

    if modularity_mode == 'uv':
        cols = ['comm_size', 'ncomms', 'size_ulist', 'ncomponents', 'commproj_size', 'commprojrel_size']
    else:
        cols = ['rcomm_size', 'rncomms', 'rncomponents', 'rcommid', 'rcommrel_size']
    cols = ['{0}{1}'.format(k, suff) for k in cols]

    if r_agg:
        rdata = np.concatenate(r_agg)
        dfr = pd.DataFrame(rdata, columns=[window_col, pm, *cols])
    else:
        dfr = pd.DataFrame(columns=[window_col, pm, *cols])

    return dfr


def compute_modularity_index_multipart(data, bipart_dicts, window_col=ye,
                                       window=None, verbose=False):
    """

    :param data:
            DataFrame of format:
                columns=[pm, window_col]

    :param bipart_dicts:
            [{id_n: [id_k]}, use_wosids]
    :param window_col:
            column on which to window // partition into groups
    :param window:
            numerical value of the window
    :param verbose: verbosity level
    :return:
    """

    ixs = sorted(data[window_col].unique())
    r_agg = []
    for ii in range(len(ixs)):
        if window:
            lower = max(0, ii - window + 1)
            mask = data[window_col].isin(ixs[lower:(ii + 1)])
        else:
            mask = data[window_col].isin(ixs[:(ii + 1)])

        ix = ixs[ii]
        pmids = list(data.loc[mask, pm].unique())
        pmids_ix = list(data.loc[data[window_col] == ix, pm].unique())
        if verbose:
            print('***')
            print('up, dn, ye: {0} {1}'.format(list(data[[up, dn]].iloc[0].values), ix))

        uv_agg = []

        if pmids_ix:
            for bipart_edges_dict in bipart_dicts:
                uvs_list = [(k, list(set(bipart_edges_dict[k]))) if k in bipart_edges_dict.keys()
                            else (k, []) for k in pmids]
                uv_agg.append(uvs_list)
        else:
            uv_agg = []
        if verbose:
            print('lens of factions {0}'.format([len(x) for x in uv_agg]))

        commsize_ncomm_usize = compute_comm_structure_reduced_graph(uv_agg, multi=True, verbose=verbose)
        csize_dict = dict(zip(pmids, commsize_ncomm_usize))
        output = [(ix, p, *csize_dict[p]) for p in pmids]
        r_agg.append(output)
    cols = ['rcomm_size', 'rncomms', 'rncomponents', 'rcommid', 'rcommrel_size']
    if window:
        suff = '{0}'.format(window)
    else:
        suff = ''

    cols = ['{0}{1}'.format(k, suff) for k in cols]

    if r_agg:
        rdata = np.concatenate(r_agg)
        dfr = pd.DataFrame(rdata, columns=[window_col, pm, *cols])
    else:
        dfr = pd.DataFrame(columns=[window_col, pm, *cols])
    return dfr


def compute_vdegrees(data):
    """
    compute the support index for a bigraph,
    as a matter of definition, for bigraphs with |U| < 2 and |V| == 0, alpha = 0
    :param data: list of lists, representing edges of a bipartite graph (U, V, E)
        the index of the list represents a vertex from set U, while a vertex from set V can be an object
    :return: (support_index, power of set V)
    """

    flat = [x for sublist in data for x in sublist]
    v_set = set(flat)
    power_v = len(v_set)

    v2i = {v: i for v, i in zip(list(v_set), range(power_v))}
    flat_int = [v2i[v] for v in flat]
    uniques, counts = np.unique(flat_int, return_counts=True)
    return uniques, counts, v2i


def compute_support(data, frac_important=0.2, mode='square'):
    """
    compute the support index for a bigraph,
    as a matter of definition, for bigraphs with |U| < 2 and |V| == 0, alpha = 0
    :param data: list of lists, representing edges of a bipartite graph (U, V, E)
        the index of the list represents a vertex from set U, while a vertex from set V can be an object
    :param frac_important:
            fraction of the important papers
    :param mode:
    :return: (support_index, power of set V)
    """

    if mode == 'square':
        def foo(x): return x**2
    elif mode == 'linear':
        def foo(x): return x
    else:
        def foo(x): return x

    uniques, counts, _ = compute_vdegrees(data)
    if len(uniques) == 0 or len(data) == 1:
        return 0, len(uniques), sum(counts)
    power_v = len(uniques)
    n_top = int(np.ceil(len(uniques)*frac_important))
    volume = n_top * foo(len(data))
    counts_sorted = sorted(counts)[::-1][:n_top]
    alpha = sum([foo(z) for z in counts_sorted]) / volume

    return alpha, power_v, sum(counts)


def compute_affinity(data):
    """
    compute the support index for a bigraph,
    as a matter of definition, for bigraphs with |U| < 2 and |V| == 0, alpha = 0
    :param data: list of lists, representing edges of a bipartite graph (U, V, E)
        the index of the list represents a vertex from set U, while a vertex from set V can be an object
    :return: (support_index, power of set V)
    """
    uniques, counts, v2i = compute_vdegrees(data)

    v_degrees = dict(zip(uniques, counts))
    n_edges = sum(counts)
    affinities = []
    for item in data:
        affinity_unnormed = sum([v_degrees[v2i[k]] - 1 for k in item])
        if len(data) == 1 or len(item) == 0:
            # or n_edges == len(item)?
            affinity = 0
        else:
            affinity = affinity_unnormed/(len(item) * (len(data) - 1))
        affinities.append(affinity)
    return affinities


def compute_comm_structure_bigraph(uvs_list, disjoint_uv=True, verbose=False):
    """

    :param uvs_list: list of u vertices
    :param disjoint_uv:
    :param verbose:
    :return: (|comm size|, |number of comms|, |U|, |number of components of G|)
    """

    ulist = [x for x, _ in uvs_list]
    vlist_flat = [x for _, sublist in uvs_list for x in sublist]
    if len(vlist_flat) > 0:
        vset = set(vlist_flat)
        if verbose:
            print('diff: {0}, len ulist: {1}, len vset {2}, len edges {3}'.format(len(vlist_flat)
                                                                                  - len(vset), len(ulist),
                                                                                  len(vset), len(vlist_flat)))

        uset_conv = {k: j for k, j in zip(ulist, range(len(ulist)))}
        # uset_conv_inv = {v: k for k, v in uset_conv.items()}
        if disjoint_uv:
            # treat U and V as disjoint sets
            vset_conv = {k: j + len(ulist) for k, j in zip(vset, range(len(vset)))}
        else:
            vset_outstanding = list(vset - set(ulist))
            vset_intersection = list(vset & set(ulist))
            vset_conv = {k: j + len(ulist) for k, j in zip(vset_outstanding, range(len(vset_outstanding)))}
            vset_conv = {**vset_conv, **{k: uset_conv[k] for k in vset_intersection}}

        edges_agg = [[(uset_conv[u], vset_conv[v]) for v in vs] for u, vs in uvs_list]
        edges_flat = [e for sublist in edges_agg for e in sublist]
        random.seed(13)
        g = ig.Graph(edges_flat, directed=False)
        communities = g.community_infomap()
        comm_df = pd.DataFrame(communities.membership,
                               index=[v.index for v in g.vs], columns=['comm_id'])
        comm_size_dict = comm_df['comm_id'].value_counts().to_dict()
        # communities restricted to u set
        uset_comm_size_dict = comm_df.loc[uset_conv.values(), 'comm_id'].value_counts().to_dict()
        if verbose:
            print('comm_size_dict', comm_size_dict)
            # print('comm_df', comm_df)
            print('comm_df len', comm_df.shape)
            print('comm_df uset len', comm_df.loc[uset_conv.values()].shape)
        uset_comm_dict = comm_df.loc[list(uset_conv.values())]['comm_id'].to_dict()
        # map: (original uset id) -> (int uset id) -> (community id) -> (community size)
        commsize_ncomm_usize = [(comm_size_dict[uset_comm_dict[uset_conv[u]]],
                                 len(communities),
                                 len(ulist),
                                 len(g.clusters()),
                                 uset_comm_size_dict[uset_comm_dict[uset_conv[u]]],
                                 uset_comm_size_dict[uset_comm_dict[uset_conv[u]]]/len(ulist))
                                for u in ulist]
    else:
        commsize_ncomm_usize = [(1, len(ulist), len(ulist), len(ulist), 1, 1) for u in ulist]
    return commsize_ncomm_usize


def bipart_graph_to_weights(uvs):
    c = combinations(uvs, 2)
    u_edges = {}

    for a_node, b_node in c:
        a, a_vs = a_node
        b, b_vs = b_node
        a_vs_set = set(a_vs)
        b_vs_set = set(b_vs)
        union = a_vs_set | b_vs_set
        if union:
            weight = len(a_vs_set & b_vs_set) / len(a_vs_set | b_vs_set)
        else:
            weight = 0
        u_edges[(a, b)] = weight
    return u_edges


def compute_comm_structure_reduced_graph(uv_list, multi=False, verbose=False):
    """

    :param uvs:
    :param multi: False is uvs is a list [u, {v}] (bipart), True if uvs is list of those [[u, {v}]] (multipart)
    :param verbose:
    :return:
    """
    # check lens of all uv_list
    uvs = uv_list if multi else [uv_list]
    # uvs = uv_list
    if verbose:
        print('in compute_comm_structure_reduced_graph() - uvs:')
        print(uvs)
    u_edges_list = [bipart_graph_to_weights(uv) for uv in uvs]
    # if ulist
    ulist = set.union(*[set([u for u, vs in item]) for item in uvs])
    superset_edges = sorted(set.union(*[set(item.keys()) for item in u_edges_list]))
    weights_matrix = [[item[k] if k in item.keys() else 0. for k in superset_edges] for item in u_edges_list]
    weights = np.array(weights_matrix).sum(axis=0)
    if superset_edges:
        max_weight = weights.max()
        if max_weight:
            weights /= max_weight
        # u : uproj
        uset_conv = {k: j for k, j in zip(ulist, range(len(ulist)))}
        uset_conv_inv = {v: k for k, v in uset_conv.items()}
        edges = [(uset_conv[x], uset_conv[y]) for x, y in superset_edges]
        g = ig.Graph(edges, directed=False)
        communities = g.community_infomap(edge_weights=weights)
        comm_df = pd.DataFrame(communities.membership,
                               index=[v.index for v in g.vs], columns=['comm_id'])
        # comm_id : size
        comm_size_dict = comm_df['comm_id'].value_counts().to_dict()
        # u_proj : comm_id
        uset_comm_dict = comm_df.loc[list(uset_conv.values())]['comm_id'].to_dict()
        if verbose:
            print('u vertices: {0}'.format(ulist))
            print('super set edges: {0}'.format(superset_edges))
            print('weights: {0}'.format(weights))
            print('weights_matrix: {0}'.format(weights_matrix))
            print('uset_conv: {0}'.format(uset_conv))
            print('uset_conv_inv: {0}'.format(uset_conv_inv))
            print('comm_size_dict', comm_size_dict)
            print('comm_df len', comm_df.shape)

        commsize_ncomm_usize = [(comm_size_dict[uset_comm_dict[uset_conv[u]]],
                                 len(communities),
                                 len(g.clusters()),
                                 uset_comm_dict[uset_conv[u]],
                                 comm_size_dict[uset_comm_dict[uset_conv[u]]] / len(ulist))
                                for u in ulist]
    else:
        commsize_ncomm_usize = [(1,
                                 len(ulist),
                                 len(ulist),
                                 0,
                                 1 / len(ulist)) for u in ulist]
    return commsize_ncomm_usize


def calculate_batch_numsum(data, bipart_edges_dict, verbose=False):
    vn = 'vnode'

    slist = data.apply(lambda x: [(x[pm], x[ye], y) for y in
                                  (bipart_edges_dict[x[pm]] if x[pm] in bipart_edges_dict.keys() else [])], axis=1)
    # list [(paper, year, [vnodes])]
    flat_list = [x for sublist in slist for x in sublist]
    if verbose:
        print(data.shape)
    if verbose:
        print(flat_list)
    if flat_list:
        uv_edges_df = pd.DataFrame(flat_list, columns=[pm, ye, vn])
        uv_year = uv_edges_df.groupby([vn, ye]).apply(lambda x: x.shape[0])
        uv_year_cumsum = uv_year.groupby(level=0).apply(lambda x: np.cumsum(x))
        uv_year_cumsum_observed = uv_year_cumsum.groupby(level=0).apply(lambda x: x.shift())
        if verbose:
            print('***')
            print(uv_year_cumsum_observed.shape)
        uv_year_cumsum_observed.loc[~uv_year_cumsum_observed.notnull()] = 0.0
        uv_year_cumsum_observed.name = 'count'
        uv_year_cumsum_observed = uv_year_cumsum_observed.reset_index()
        if verbose:
            print('***')
            print(uv_edges_df.head())
            print(uv_year_cumsum_observed.head())

        ultimate_df = pd.merge(uv_edges_df, uv_year_cumsum_observed, on=[vn, ye], how='inner')
        take_max = ultimate_df.groupby([pm, ye]).apply(lambda x: x['count'].max())
        take_max.name = 'count'
        take_max = take_max.reset_index()
    else:
        take_max = pd.DataFrame([], columns=[pm, 'count'])
    pm_absent = list(set(data[pm]) - set(bipart_edges_dict.keys()))
    if verbose:
        print(len(pm_absent))
    take_max_absent = pd.DataFrame([(p, 0.) for p in pm_absent], columns=[pm, 'count'])
    ultimate_df = pd.concat([take_max[[pm, 'count']], take_max_absent])
    return ultimate_df[[pm, 'count']]
    # return take_max, take_max_absent