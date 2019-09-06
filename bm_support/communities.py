from datahelpers.constants import iden, ye, ai, ps, up, dn, ar, ni, cexp, \
    qcexp, nw, wi, dist, rdist, bdist, pm, cpop, cden, ct, affs, aus


def pick_by_community(df, feature='authors', how=1):
    comm_id = f'{feature}_rcommid'
    cs = ['_rcomm_size', '_rcommrel_size']
    cs = [f'{feature}{x}' for x in cs]
    df2 = df.copy()
    flags = df2.groupby([up, dn]).apply(lambda x: x[comm_id].unique().shape[0] > 1)
    if isinstance(how, int):
        # pick interactions with more than one community
        interactions_with_multiple_communities = flags[flags].reset_index()
        df2 = df2.merge(interactions_with_multiple_communities, on=[up, dn], how='right')
        #
        small_comm = df2.groupby([up, dn, ye], as_index=False).apply(lambda x: x.loc[x[comm_id] == how])
    elif how == 'mono':
        # pick interactions with only one community
        interactions_with_multiple_communities = flags[~flags].reset_index()
        small_comm = df2.merge(interactions_with_multiple_communities, on=[up, dn], how='right')
    return small_comm
