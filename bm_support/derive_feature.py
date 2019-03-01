import numpy as np
import pandas as pd
from datahelpers.constants import up, dn, pm, ye
from scipy.stats import norm

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
