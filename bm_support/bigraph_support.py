import pandas as pd
import numpy as np
from datahelpers.constants import iden, ye, ai, ps, up, dn, ar, ni, cexp, qcexp, dist, rdist, pm


def compute_support_index(data, bipart_edges_dict, pm_wid_dict, window_col=ye,
                          window=None, frac_important=0.1,
                          transform='square', mode='all', use_wosids=True):
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
            cur_wosids = cur_pmids
        if mode == 'all':
            cur_cites = [bipart_edges_dict[k] for k in cur_wosids]
        elif mode == 'cross':
            cur_cites = [[y for y in bipart_edges_dict[k] if y in cur_wosids] for k in cur_wosids]

        # ll = len([x for sublist in cur_cites for x in sublist])
        # print(len(cur_cites), ll, len(cur_wosids), skip_wosids)
        # print(cur_wosids, cur_cites)
        alpha, n_unique_citations, n_edges = compute_support(cur_cites, frac_important, transform)
        ind_agg.append(ix)
        r_agg.append((alpha, len(cur_cites), n_unique_citations, n_edges))

    if window:
        suff = '{0}'.format(window)
    else:
        suff = ''
    cols = ['suppind', 'size_level_a', 'size_level_b', 'n_edges']
    # ren_cols = {k: '{0}{1}'.format(k, suff) for k in cols}
    ren_cols = ['{0}{1}'.format(k, suff) for k in cols]
    # print(ren_cols)
    dfr = pd.DataFrame(r_agg, index=ind_agg, columns=ren_cols)
    dfr = dfr.rename_axis(window_col, axis='index')
    return dfr


def compute_affinity_index(data, bipart_edges_dict, pm_wid_dict, window_col=ye,
                           window=None, mode='all', use_wosids=True):
    """
    #TODO this function has to be double checked, the filters might behave unpredictably
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

        cur_pmids = list(data.loc[mask, pm].unique())
        pmids_ix = [pmx for pmx in data.loc[data[window_col] == ix, pm].unique() if pmx in pm_wid_dict.keys()]
        if use_wosids:
            cur_pmids_present = [k for k in cur_pmids if k in pm_wid_dict.keys()]
            cur_wosids = [pm_wid_dict[k] for k in cur_pmids_present]
        else:
            cur_wosids = cur_pmids
            cur_pmids_present = cur_pmids
        if cur_pmids_present:
            if mode == 'cross':
                cur_cites = [[y for y in bipart_edges_dict[k] if y in cur_wosids] for k in cur_wosids]
            else:
                cur_cites = [bipart_edges_dict[k] for k in cur_wosids]
            alphas = compute_affinity(cur_cites)
            alphas_dict = dict(zip(cur_pmids_present, alphas))
            output = [(ix, p, alphas_dict[p]) for p in pmids_ix]
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
    mean_degree_v = np.mean([len(x) for x in data])

    n_top = int(np.ceil(frac_important * mean_degree_v))
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
