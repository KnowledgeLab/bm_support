from datahelpers.constants import iden, ye, ai, ps, up, dn, ar, ni, cexp, \
    qcexp, nw, wi, dist, rdist, bdist, pm, cpop, cden, ct, affs, aus


def pick_by_community(df, feature='authors', window=None, how=1):

    if window:
        suff = f'{window}'
    else:
        suff = ''

    commid_col = f'rcommid{suff}'
    rank_col = f'rank{suff}'

    df2 = df.copy()
    flags = df2.groupby([up, dn]).apply(lambda x: x[commid_col].unique().shape[0] > 1)
    if isinstance(how, int):
        # pick interactions with more than one community
        interactions_with_multiple_communities = flags[flags].reset_index()
        df2 = df2.merge(interactions_with_multiple_communities, on=[up, dn], how='right')

        rank_commids = df2.groupby([up, dn, ye]).apply(lambda x: x.loc[x[rank_col] == how, commid_col]).reset_index()
        # trim communities that are ranked the same but have different ids
        # print(rank_commids.head())
        rank_commids = rank_commids.drop_duplicates([up, dn, ye])
        dfr = df2.merge(rank_commids, on=[up, dn, ye, commid_col])
    # elif how == 'mono':
    #     # pick interactions with only one community
    #     interactions_with_multiple_communities = flags[~flags].reset_index()
    #     small_comm = df2.merge(interactions_with_multiple_communities, on=[up, dn], how='right')
    return dfr
