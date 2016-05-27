import pandas as pd
import datahelpers.dftools as dfto
import datahelpers.collapse as dc


def transform_df(dfi, statement_columns, action_column, claim_column,
                 index_name, aggregate_negs=True):
    df = dfi.copy()
    print 'df shape:', df.shape, '. Type of', claim_column, \
        'column', df[claim_column].dtype
    index_cs = statement_columns + [action_column]
    n_unique_triplets = df.drop_duplicates(index_cs).shape[0]
    if aggregate_negs:
        df = dc.aggregate_negatives_boolean_style(df, statement_columns,
                                                  action_column, claim_column)
        print 'Number of unique (i_a, i_b, at) triplets after aggregation:', \
            df.drop_duplicates(index_cs).shape[0]
        print 'Fraction of unique (i_a, i_b, at) '\
            'triplets remaining after aggregation:', \
            df.drop_duplicates(index_cs).shape[0]/float(n_unique_triplets)
    # create new integer index for (i_a, i_b, at)
    df = dfto.get_multiplet_to_int_index(df, index_cs, index_name)
    return df


def try_impute_else_discard(df, impute_column, grouping_column):
    working_columns = [grouping_column, impute_column]
    df2 = df.copy()
    m = df2[impute_column].isnull()
    print 'Number of NAs:', sum(m), 'values, total size', df2.shape[0]
    df2[impute_column] = df[working_columns].groupby(grouping_column,
                                          as_index=False).transform(lambda g:
                                                                    g.fillna(g.mean()))[impute_column]
    m = df2[impute_column].isnull()
    print 'Could not impute', sum(m), \
        'values (discarded), out of ', df2.shape[0]
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