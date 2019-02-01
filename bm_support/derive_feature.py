import numpy as np
import pandas as pd
from datahelpers.constants import up, dn, pm, ye


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
    flags = [(False, False)] + [(u * v < 0, abs(v) < abs(u)) for v, u in zip(s.values[1:], s.values[:-1])]
    r = pd.DataFrame(flags, index=item[time_col], columns=['sign_change', 'decrease'])
    return r
