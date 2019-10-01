from datahelpers.constants import iden, ye, ai, ps, up, dn


def pick_community_by_rank(df, window=None, comm_rank=1, min_comms=2):

    if window:
        suff = f'{window}'
    else:
        suff = ''

    commid_col = f'rcommid{suff}'
    rank_col = f'rank{suff}'

    df2 = pick_min_communities(df, window, min_comms)

    rank_commids = df2.groupby([up, dn]).apply(lambda x: x.loc[x[rank_col] == comm_rank,
                                                                commid_col]).reset_index()
    # trim communities that are ranked the same but have different ids
    rank_commids = rank_commids.drop_duplicates([up, dn])
    dfr = df2.merge(rank_commids, on=[up, dn, commid_col])
    return dfr


def pick_min_communities(df, window=None, min_comm_sizes=2):

    if window:
        suff = f'{window}'
    else:
        suff = ''

    commid_col = f'rcommid{suff}'
    rank_col = f'rank{suff}'

    min_ranks_composition = list(range(min_comm_sizes))
    dfr = df.copy()
    flags = dfr.groupby([up, dn]).apply(lambda x:
                                        sorted(x[rank_col].unique())[:min_comm_sizes] == min_ranks_composition)
    interactions_with_multiple_communities = flags[flags].reset_index()
    dfr = dfr.merge(interactions_with_multiple_communities, on=[up, dn], how='right')
    return dfr


def pick_interval_communities(df, window=None, interval=(1, 1)):

    a, b = interval

    if window:
        suff = f'{window}'
    else:
        suff = ''

    commid_col = f'rcommid{suff}'
    ncomms = f'rncomms{suff}'

    flags = (df[ncomms] >= a) & (df[ncomms] <= b)
    dfr = df.loc[flags].copy()
    return dfr


def pick_interval_communities_alt(df, window=None, interval=(1, 1)):

    a, b = interval

    if window:
        suff = f'{window}'
    else:
        suff = ''

    commid_col = f'rcommid{suff}'
    ncomms = f'rncomms{suff}'

    comm_size = df.groupby([up, dn]).apply(lambda x: len(x[commid_col].unique()))

    flags = (comm_size >= a) & (comm_size <= b)
    updns = flags[flags].reset_index()
    dfr = updns.merge(df, on=[up, dn])
    return dfr
