import numpy as np
from datahelpers.constants import up, dn, pm, ye, ps, cexp, ye
from scipy.stats import norm
from os.path import expanduser, join
import pandas as pd

# TODO add exponential average


def add_t0_flag(df):
    t0_mask = (
        df.drop_duplicates([up, dn, ye])
        .groupby([up, dn])
        .apply(lambda x: x.loc[x[ye].idxmin(), ye])
    )
    t0_mask = t0_mask.reset_index().rename(columns={0: ye})
    t0_mask["t0_flag"] = True
    df = df.merge(t0_mask, on=[up, dn, ye], how="left")
    df.loc[df["t0_flag"].isnull(), "t0_flag"] = False
    return df


def moving_average_windows(means, sizes, index, column_name, windows=(None, 1)):
    positives = [m * s for m, s in zip(means, sizes)]
    return moving_average_windows_(positives, sizes, index, column_name, windows)


def moving_average_windows_(npositives, sizes, index, column_name, windows=(None, 1)):

    """
    compute the moving average
    :param npositives:
    :param sizes:
    :param index:
    :param column_name:
    :param windows:
    :return:
    moving_average_windows([1, 2, 3], [4, 4, 4], [190, 191, 192], 'cc', [None, 2])
                 cc_ma_None  cc_ma_2
        190         NaN      NaN
        191       0.250    0.250
        192       0.375    0.375
    """
    r_series = []
    for window in windows:
        if window is not None:
            window_inds = [(max(k - window, 0), k) for k in range(1, len(sizes))]
            est_mus = [sum(npositives[i:j]) / sum(sizes[i:j]) for i, j in window_inds]
        else:
            est_mus = [
                sum(npositives[:j]) / sum(sizes[:j]) for j in range(1, len(sizes))
            ]
        s = pd.Series(
            [np.nan] + est_mus,
            index=index,
            name="{0}_ma_{1}".format(column_name, window),
        )
        r_series.append(s)
    return pd.concat(r_series, axis=1)


def assign_positive_transition_flag(
    item, transition_batch_flag="good_transition", flag_name="gt_claim"
):
    # relies on the fact that there are only 2 rdist values
    # gt_claim ~ good transition claim
    mask = item["rdist"].abs() < 0.5
    mask_gt = item[transition_batch_flag]
    r = item[[up, dn, pm]].copy()
    r[flag_name] = mask & mask_gt
    r = r
    return r


def find_transition(item, time_col=ye, value_col="mean_rdist"):
    # value_col should be sorted wrt to time_col
    s = item[value_col]
    flags = [(False, False, 0, 0, np.nan)] + [
        (u * v < 0, abs(v) < abs(u), np.sign(abs(v) - abs(u)), abs(v) - abs(u), u)
        for v, u in zip(s.values[1:], s.values[:-1])
    ]

    columns = ["sign_change", "decrease", "sign_diff_abs", "diff_abs", "prev"]
    columns_ = ["{0}_{1}".format(c, value_col) for c in columns]
    r = pd.DataFrame(flags, index=item[time_col], columns=columns_)

    return r


def ppf_smart(x, epsilon=1e-6):
    if x < epsilon:
        return norm.ppf(epsilon)
    elif x > 1 - epsilon:
        return norm.ppf(1 - epsilon)
    else:
        return norm.ppf(x)


def attach_moving_averages_per_interaction(
    df0, correct_column, claim_col, interaction_column, windows=[None, 2]
):

    kn_int_per_year = df0.groupby([up, dn, ye]).apply(
        lambda x: pd.Series([x[correct_column].sum(), x.shape[0]], index=["k", "n"])
    )
    ma_per_int = kn_int_per_year.groupby(level=[0, 1]).apply(
        lambda batch: moving_average_windows_(
            batch.k, batch.n, batch.index, correct_column, windows
        )
    )

    # result | index: *interaction_index, year; columns : *ma_averages of correctness distance

    kn_per_year = df0.groupby([ye], sort=True).apply(
        lambda x: pd.Series(
            [x[claim_col].sum(), x.shape[0]], index=["k_" + claim_col, "n_" + claim_col]
        )
    )
    ma_claim_time = moving_average_windows_(
        kn_per_year["k_" + claim_col],
        kn_per_year["n_" + claim_col],
        kn_per_year.index,
        claim_col,
        windows,
    )

    # fill the first year with something mildly positive
    ma_claim_time.iloc[0] = [0.7, 0.3]
    claim_ma_columns = ma_claim_time.columns

    # result | index: year; columns: *ma_averages or claims

    t0_int = (
        df0.drop_duplicates([up, dn, ye])
        .groupby([up, dn])
        .apply(lambda x: x.loc[x[ye].idxmin(), [ye, interaction_column]])
    )

    # result| index: *interaction_index; columns: year; interaction_sign

    ma_claim_t0_int = pd.merge(
        t0_int, ma_claim_time, left_on=ye, right_index=True, how="left"
    )

    for c in claim_ma_columns:
        new_column_name = "bdist_" + "_".join(c.split("_")[1:])
        ma_claim_t0_int[new_column_name] = 1.0 - ma_claim_t0_int[c]
        mask_incorrect = (
            (ma_claim_t0_int[interaction_column] == 1) & (ma_claim_t0_int[c] >= 0.5)
        ) | ((ma_claim_t0_int[interaction_column] == 0) & (ma_claim_t0_int[c] < 0.5))
        ma_claim_t0_int.loc[mask_incorrect, new_column_name] = ma_claim_t0_int[c]
        # if interaction is negative (=1) and ma claim columns is positive (>=0.5) or ...

    ma_claim_t0_int2 = ma_claim_t0_int.reset_index().set_index([up, dn, ye])
    ma_per_int.update(ma_claim_t0_int2)

    return ma_per_int


def attach_transition_metrics(df, target_col):
    # derive sign change variable
    # one year long histories are filtered out

    mns = df.groupby([up, dn, ye]).apply(
        lambda x: pd.Series(x[target_col].mean(), index=[target_col])
    )

    long_flag = mns.groupby(level=[0, 1], group_keys=False).apply(
        lambda x: x.shape[0] > 1
    )

    mns_long = mns.loc[long_flag].reset_index().sort_values([up, dn, ye])

    change_flags = (
        mns_long.groupby([up, dn])
        .apply(lambda x: find_transition(x, value_col=target_col))
        .reset_index()
    )

    dfn = pd.merge(df, change_flags, on=[up, dn, ye], how="right").sort_values(
        [up, dn, ye]
    )
    return dfn


def select_t0(df, t0=True, acolumns=(up, dn), t_column=ye):

    acolumns = list(acolumns)
    all_cols = acolumns + [t_column]
    dfz = df[all_cols].drop_duplicates(all_cols).sort_values(all_cols)
    dfz = dfz.groupby(acolumns).apply(
        lambda x: pd.DataFrame(
            index=(x[ye].values[:1] if t0 else x[ye].values[1:])
        ).rename_axis(t_column)
    )
    dfz = dfz.reset_index()
    df2 = pd.merge(df, dfz, how="inner", on=all_cols)
    return df2


def expand_bounded_vector(s, eps=1e-6):

    a = np.min(s)
    b = np.max(s)
    slope = (1 - 2.0 * eps) / (b - a)
    const = eps - slope * a
    s = s.apply(lambda x: x * slope + const)
    s = s.apply(lambda x: np.log(x / (1 - x)))
    return s


def derive_refutes_community_features(
    df,
    fpath,
    windows=(2, 3, None),
    metric_sources=("past", "authors", "affiliations"),
    work_var=ps,
    verbose=False,
):
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
    mt = "redmodularity"
    df_agg = []
    for ws in windows:
        for ms in metric_sources:
            if verbose:
                print("window {0}, metrics source {1}".format(ws, ms))
            # work_var average or popular vote (work_var flag)
            comm_ave_col = (
                "{0}_comm_ave_{1}{2}".format(work_var, ms, ws)
                if ws
                else "{0}_comm_ave_{1}".format(work_var, ms)
            )

            fn = expanduser("{0}{1}_metric_{2}_w{3}.csv.gz".format(fpath, mt, ms, ws))
            # load community df
            dfc = pd.read_csv(fn, index_col=0)

            if verbose:
                print("dfc items {0}".format(dfc.shape[0]))

            # community id column name in dfc
            commid_col = "rcommid{0}".format(ws) if ws else "rcommid"
            # new community id column name
            gen_commid_col = "{0}_{1}".format(ms, commid_col)

            # we prefer gen_commid_col : when merging it should have traces of window size and metrics source
            # which the original dfc does not have
            dfc = dfc.rename(columns={commid_col: gen_commid_col})

            # merge the work variable (work_var) onto community data
            dfc2 = dfc.merge(dfw[[up, dn, pm, work_var]], on=[up, dn, pm], how="left")

            # filter out NA work_var
            dfc3 = dfc2.loc[dfc2[work_var].notnull()]

            # filter out up, dn, ylook, gen_commid_col for which there are not communities y < ylook
            dfc3_flag = dfc3.groupby([up, dn, "ylook", gen_commid_col]).apply(
                lambda x: (x[ye] < x["ylook"]).sum() > 0
            )
            good_batches = dfc3_flag.loc[dfc3_flag].reset_index()
            dfc4 = pd.merge(dfc3, good_batches, on=[up, dn, "ylook", gen_commid_col])
            # calculate work_var average or popular vote (work_var flag)
            # per interaction (up, dn), year, community
            df_mean_per_comm = (
                dfc4.groupby([up, dn, "ylook", gen_commid_col])
                .apply(lambda x: int(x.loc[x[ye] < x["ylook"]][work_var].mean() > 0.5))
                .reset_index()
            )

            # name work_var average or popular vote (work_var flag) column
            df_mean_per_comm = df_mean_per_comm.rename(columns={0: comm_ave_col})

            # filter not null work_var average
            df_mean_per_comm2 = df_mean_per_comm[
                df_mean_per_comm[comm_ave_col].notnull()
            ]

            # for each group (up, dn, t) - claims made in (t - window, t] keep only claims made at t
            dfc_claims = dfc.loc[dfc[ye] == dfc["ylook"]]

            # merge work_var average onto available claims by up, dn, year, community
            df_metric_claims = pd.merge(
                dfc_claims[[up, dn, ye, gen_commid_col, pm]],
                df_mean_per_comm2[[up, dn, "ylook", gen_commid_col, comm_ave_col]],
                left_on=[up, dn, ye, gen_commid_col],
                right_on=[up, dn, "ylook", gen_commid_col],
                how="left",
            )

            # now trim df_metric_claims : drop NAs
            df_ = df_metric_claims.loc[
                df_metric_claims[comm_ave_col].notnull(), [up, dn, pm, comm_ave_col]
            ].copy()

            # now trim df_metric_claims : drop claims not from dfw
            df_ = pd.merge(dfw[[up, dn, pm]], df_, on=[up, dn, pm], how="left")
            if verbose:
                print("df_ items {0}".format(df_.shape[0]))

            df_agg.append(df_)
            # # merge work_var average onto publications by up, dn, publication
    # dfr = pd.concat(df_agg)
    dfm = df_agg[0].copy()
    for df_ in df_agg[1:]:
        dfm = dfm.merge(df_, on=[up, dn, pm])
    return dfm


def clean_nas(df, feats, verbose=False):
    masks = []
    cfeats = list(set(feats).intersection(set(df.columns)))
    for c in cfeats:
        mask = df[c].notnull()
        masks.append(mask)
        if verbose:
            if sum(mask) != mask.shape[0]:
                print(c, sum(mask))

    mask_agg = masks[0]
    for m in masks[1:]:
        mask_agg &= m
    dfr = df.loc[mask_agg].copy()
    return dfr
