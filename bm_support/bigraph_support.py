import pandas as pd
import numpy as np
from datahelpers.constants import iden, ye, ai, ps, up, dn, ar, ni, cexp, qcexp, dist, rdist, pm


def compute_support_index(data, bipart_edges_dict, pm_wid_dict, window_col=ye,
                          window=2, frac_important=0.1, transform='square', mode='all'):
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
        mask = (data[window_col] <= ix) & (data[window_col] > ix - window)
        cur_pmids = data.loc[mask, pm].unique()
        cur_wosids = [pm_wid_dict[k] for k in cur_pmids if k in pm_wid_dict.keys()]
        if mode == 'all':
            cur_cites = [bipart_edges_dict[k] for k in cur_wosids]
        elif mode == 'cross':
            cur_cites = [[y for y in bipart_edges_dict[k] if y in cur_wosids] for k in cur_wosids]
        alpha, n_unique_citations = compute_support(cur_cites, frac_important, transform)
        ind_agg.append(ix)
        r_agg.append((alpha, len(cur_cites), n_unique_citations))
    dfr = pd.DataFrame(r_agg, index=ind_agg, columns=['alpha', 'n_papers', 'n_cited'])
    dfr = dfr.rename_axis(window_col, axis='index')
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
        return 0, len(uniques)
    power_v = len(uniques)
    mean_degree_v = np.mean([len(x) for x in data])

    n_top = int(np.ceil(frac_important * mean_degree_v))
    volume = n_top * foo(len(data))
    counts_sorted = sorted(counts)[::-1][:n_top]
    alpha = sum([foo(z) for z in counts_sorted]) / volume

    return alpha, power_v


def compute_affinity(data):
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
    uniques, counts, v2i = compute_vdegrees(data)

    v_degrees = dict(zip(uniques, counts))
    n_edges = sum(counts)
    affinities = []
    for item in data:
        affinity_unnormed = sum([v_degrees[v2i[k]] - 1 for k in item])
        affinity = affinity_unnormed/(n_edges - len(item))
        affinities.append(affinity)
    return affinities
