from scipy import stats
import datahelpers.dftools as dfto
import datahelpers.collapse as dc
import datahelpers.plotting as dp


def rank_degenerate_values(keys, values, sort=True):

    """
    create ranking of values

    :param keys: list of ids
    :param values: list
    :param sort: sort if keys and values are not sorted
    :return:

    example :
        keys = ['a', 'b', 'c', 'd']
        values = [5, 3, 2, 3]
    output:
        [('a', 5, 0), ('b', 3, 1), ('d', 3, 1), ('c', 2, 2)]
    """
    if sort:
        sorted_kv = sorted(zip(keys, values), key=lambda x: x[1], reverse=True)
    else:
        sorted_kv = list(zip(keys, values))
    agg = [(*sorted_kv[0], 0)]
    for k, v in sorted_kv[1:]:
        kp, vp, ip = agg[-1]
        if vp == v:
            agg.append((k, v, ip))
        else:
            agg.append((k, v, ip + 1))
    result = [(x, z) for x, y, z in agg]
    return result


def transform_df(dfi, statement_columns, action_column, claim_column,
                 index_name, aggregate_negs=True):
    df = dfi.copy()
    print('df shape:', df.shape, '. Type of', claim_column, \
        'column', df[claim_column].dtype)
    index_cs = statement_columns + [action_column]
    n_unique_triplets = df.drop_duplicates(index_cs).shape[0]
    if aggregate_negs:
        df = dc.aggregate_negatives_boolean_style(df, statement_columns,
                                                  action_column, claim_column)
        print('Number of unique (i_a, i_b, at) triplets after aggregation:',
              df.drop_duplicates(index_cs).shape[0])
        print('Fraction of unique (i_a, i_b, at) '
              'triplets remaining after aggregation:',
              df.drop_duplicates(index_cs).shape[0]/float(n_unique_triplets))
    # create new integer index for (i_a, i_b, at)
    df = dfto.get_multiplet_to_int_index(df, index_cs, index_name)
    return df


def try_impute_else_discard(df, impute_column, grouping_column):
    working_columns = [grouping_column, impute_column]
    df2 = df.copy()
    m = df2[impute_column].isnull()
    print('Number of NAs:', sum(m), 'values, total size', df2.shape[0])
    df2[impute_column] = df[working_columns].groupby(grouping_column,
                                                     as_index=False).transform(lambda g:
                                                                    g.fillna(g.mean()))[impute_column]
    m = df2[impute_column].isnull()
    print('Could not impute', sum(m),
          'values (discarded), out of ', df2.shape[0])
    return df2.loc[~m]


def define_guess_quality(df, c_gt, c_ps, c_gu):
    df[c_gu] = dfto.XOR(df[c_gt], df[c_ps])
    return df


def pdf_cut(df, c, value_cutoff):
    dfr = df.loc[df[c] > value_cutoff].copy()
    return dfr


def pdf_cut_func(df, c, value_cutoff, func):
    dfr = df.loc[df[c].apply(lambda x: func(x, value_cutoff))].copy()
    return dfr


def split_data(df, c):
    m = (df[c] == 0)
    data_ps = df.loc[~m].values[:, :]
    data_ng = df.loc[m].values[:, :]
    vals = [data_ps, data_ng]
    return vals


def ks_and_hist(pair, k):
    print(stats.ks_2samp(pair[0][:, k], pair[1][:, k]))
    pl_info = dp.plot_hist([pair[0][:, k], pair[1][:, k]],
                           approx_nbins=5, opacity=0.5,
                           ylog_axis=True, normed_flag=True)
