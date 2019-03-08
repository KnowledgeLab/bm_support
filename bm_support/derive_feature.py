import numpy as np
import pandas as pd
from datahelpers.constants import up, dn, pm, ye
from scipy.stats import norm
from os.path import expanduser
from datahelpers.constants import ps


# TODO add exponential average


def moving_average_windows(means, sizes, index, column_name, windows=(None, 1)):

    """
    compute the moving average
    :param means:
    :param sizes:
    :param index:
    :param column_name:
    :param windows:
    :return:
    """
    r_series = []
    for window in windows:
        rprods = [s * rmu for rmu, s in zip(means, sizes)]
        if window is not None:
            window_inds = [(max(k - window, 0), k) for k in range(1, len(sizes))]
            est_mus = [sum(rprods[i:j]) / sum(sizes[i:j]) for i, j in window_inds]
        else:
            est_mus = [sum(rprods[:j]) / sum(sizes[:j]) for j in range(1, len(sizes))]
        s = pd.Series([np.nan] + est_mus, index=index,
                      name='{0}_ma_{1}'.format(column_name, window))
        r_series.append(s)
    return pd.concat(r_series, axis=1)


def assign_positive_transition_flag(item, transition_batch_flag='good_transition', flag_name='gt_claim'):
    # relies on the fact that there are only 2 rdist values
    # gt_claim ~ good transition claim
    mask = (item['rdist'].abs() < 0.5)
    mask_gt = item[transition_batch_flag]
    r = item[[up, dn, pm]].copy()
    r[flag_name] = mask & mask_gt
    r = r
    return r


def find_transition(item, time_col=ye, value_col='mean_rdist'):
    # value_col should be sorted wrt to time_col
    s = item[value_col]
    flags = [(False, False, 0, 0, np.nan)] + \
            [(u * v < 0, abs(v) < abs(u), np.sign(abs(v) - abs(u)), abs(v) - abs(u), u)
             for v, u in zip(s.values[1:], s.values[:-1])]

    columns = ['sign_change', 'decrease', 'sign_diff_abs', 'diff_abs', 'prev']
    columns_ = ['{0}_{1}'.format(c, value_col) for c in columns]
    r = pd.DataFrame(flags, index=item[time_col], columns=columns_)

    return r


def ppf_smart(x, epsilon=1e-6):
    if x < epsilon:
        return norm.ppf(epsilon)
    elif x > 1 - epsilon:
        return norm.ppf(1-epsilon)
    else:
        return norm.ppf(x)


def attach_transition_metrics(df, target_col):
    # derive sign change variable
    # one year long histories are filtered out

    mns = df.groupby([up, dn, ye]).apply(lambda x: pd.Series(x[target_col].mean(),
                                                             index=[target_col]))

    long_flag = mns.groupby(level=[0, 1], group_keys=False).apply(lambda x: x.shape[0] > 1)

    mns_long = mns.loc[long_flag].reset_index().sort_values([up, dn, ye])

    change_flags = mns_long.groupby([up, dn]).apply(lambda x: find_transition(x, value_col=target_col)).reset_index()

    dfn = pd.merge(df, change_flags, on=[up, dn, ye], how='right').sort_values([up, dn, ye])
    return dfn


def select_t0(df, t0=True, acolumns=(up, dn), t_column=ye):

    acolumns = list(acolumns)
    all_cols = acolumns + [t_column]
    dfz = df[all_cols].drop_duplicates(all_cols).sort_values(all_cols)
    dfz = dfz.groupby(acolumns).apply(lambda x: pd.DataFrame(index=(x[ye].values[:1]
                                                             if t0 else x[ye].values[1:])).rename_axis(t_column))
    dfz = dfz.reset_index()
    df2 = pd.merge(df, dfz, how='inner', on=all_cols)
    return df2


def expand_bounded_vector(s, eps=1e-6):

    a = np.min(s)
    b = np.max(s)
    slope = (1 - 2.*eps)/(b - a)
    const = eps - slope*a
    s = s.apply(lambda x: x*slope + const)
    s = s.apply(lambda x: np.log(x/(1 - x)))
    return s


def derive_refutes_community_features(df, fpath, windows=(2, 3, None),
                                      metric_sources=('past', 'authors', 'affiliations'),
                                      work_var=ps,
                                      verbose=False):
    """
    :param df: the dataframe with work_var values
    :param fpath: where community datafiles are, e.g.
               up   dn  ylook      pmid  rcomm_size  rncomms  rncomponents  \
        0   2  109   2000  10970099         1.0      1.0           1.0
        1   2  213   1972   5086222         1.0      1.0           1.0
        2   2  213   1991   1711509         1.0      2.0           1.0
        3   2  213   1991   5086222         1.0      2.0           1.0
        4   2  213   2008   1711509         1.0      3.0           1.0

                       affiliations_rcommid  rcommrel_size  year
        0                   0.0       1.000000  2000
        1                   0.0       1.000000  1972
        2                   1.0       0.500000  1991
        3                   0.0       0.500000  1972
        4                   2.0       0.333333  1991

    :param windows: communities are calculated for publications in (t - window, t] time intervals
            NB: in dfc ylook is t, while year is the t of a publication in (t - window, t]
    :param metric_sources: can be past (based of references), authors, affiliations (or future etc.)
    :param work_var:
    :param verbose:
    :return:
    """

    dfw = df.copy()
    mt = 'redmodularity'
    df_agg = []
    for ws in windows:
        for ms in metric_sources:
            if verbose:
                print('window {0}, metrics source {1}'.format(ws, ms))
            # work_var average or popular vote (work_var flag)
            comm_ave_col = '{0}_comm_ave_{1}{2}'.format(work_var, ms, ws) \
                if ws else '{0}_comm_ave_{1}'.format(work_var, ms)

            fn = expanduser('{0}{1}_metric_{2}_w{3}.csv.gz'.format(fpath, mt, ms, ws))
            # load community df
            dfc = pd.read_csv(fn, index_col=0)

            if verbose:
                print('dfc items {0}'.format(dfc.shape[0]))

            # community id column name in dfc
            commid_col = 'rcommid{0}'.format(ws) if ws else 'rcommid'
            # new community id column name
            gen_commid_col = '{0}_{1}'.format(ms, commid_col)

            # we prefer gen_commid_col : when merging it should have traces of window size and metrics source
            # which the original dfc does not have
            dfc = dfc.rename(columns={commid_col: gen_commid_col})

            # merge the work variable (work_var) onto community data
            dfc2 = dfc.merge(dfw[[up, dn, pm, work_var]], on=[up, dn, pm], how='left')

            # filter out NA work_var
            dfc3 = dfc2.loc[dfc2[work_var].notnull()]

            # filter out up, dn, ylook, gen_commid_col for which there are not communities y < ylook
            dfc3_flag = dfc3.groupby([up, dn, 'ylook', gen_commid_col]).apply(lambda x:
                                                                              (x[ye] < x['ylook']).sum() > 0)
            good_batches = dfc3_flag.loc[dfc3_flag].reset_index()
            dfc4 = pd.merge(dfc3, good_batches, on=[up, dn, 'ylook', gen_commid_col])
            # calculate work_var average or popular vote (work_var flag)
            # per interaction (up, dn), year, community
            df_mean_per_comm = dfc4.groupby([up, dn, 'ylook',
                                             gen_commid_col]).apply(lambda x:
                int(x.loc[x[ye] < x['ylook']][work_var].mean() > 0.5)).reset_index()

            # name work_var average or popular vote (work_var flag) column
            df_mean_per_comm = df_mean_per_comm.rename(columns={0: comm_ave_col})

            # filter not null work_var average
            df_mean_per_comm2 = df_mean_per_comm[df_mean_per_comm[comm_ave_col].notnull()]

            # for each group (up, dn, t) - claims made in (t - window, t] keep only claims made at t
            dfc_claims = (dfc.loc[dfc[ye] == dfc['ylook']])

            # merge work_var average onto available claims by up, dn, year, community
            df_metric_claims = pd.merge(dfc_claims[[up, dn, ye, gen_commid_col, pm]],
                                        df_mean_per_comm2[[up, dn, 'ylook', gen_commid_col, comm_ave_col]],
                                        left_on=[up, dn, ye, gen_commid_col],
                                        right_on=[up, dn, 'ylook', gen_commid_col], how='left')

            # now trim df_metric_claims : drop NAs
            df_ = df_metric_claims.loc[df_metric_claims[comm_ave_col].notnull(), [up, dn, pm, comm_ave_col]].copy()

            # now trim df_metric_claims : drop claims not from dfw
            df_ = pd.merge(dfw[[up, dn, pm]], df_, on=[up, dn, pm], how='left')
            if verbose:
                print('df_ items {0}'.format(df_.shape[0]))

            df_agg.append(df_)
            # # merge work_var average onto publications by up, dn, publication
            # dfw = dfw.merge(df_metric_claims[[up, dn, pm, comm_ave_col]], on=[up, dn, pm], how='left')
    # dfr = pd.concat(df_agg)
    dfm = df_agg[0].copy()
    for df_ in df_agg[1:]:
        dfm = dfm.merge(df_, on=[up, dn, pm])
    return dfm
