import pandas as pd
from os.path import expanduser
import numpy as np
from cvxpy import Variable, Parameter, Minimize, Problem
from cvxpy import multiply as cvx_mul
from cvxpy import sum as cvx_sum
from scipy.stats import beta

up, dn, ps, pm, cexp = "up", "dn", "pos", "pmid", "cdf_exp"


def get_wdist(f1, f2, grid):
    pk_ = np.array([f1(x) for x in grid])
    qk_ = np.array([f2(x) for x in grid])
    pk_ = pk_ / np.sum(pk_)
    qk_ = qk_ / np.sum(qk_)
    dist = np.zeros((pk_.size, qk_.size))
    for i in range(dist.shape[0]):
        for j in range(dist.shape[1]):
            dist[i, j] = abs(i - j) / pk_.size
    mx = pk_ / np.sum(pk_)
    my = qk_ / np.sum(qk_)
    plan = Variable((pk_.size, qk_.size))
    mx_trans = mx.reshape(-1, 1) * dist
    mu_trans_param = Parameter(mx_trans.shape, value=mx_trans)
    obj = Minimize(cvx_sum(cvx_mul(plan, mu_trans_param)))
    plan_i = cvx_sum(plan, axis=1)
    my_constraint = mx * plan
    constraints = [
        my_constraint == my,
        plan >= 0,
        plan <= 1,
        plan_i == np.ones(pk_.size),
    ]
    problem = Problem(obj, constraints)
    solver = None
    solver_options = {}
    wd = problem.solve(solver=solver, **solver_options)
    return wd


origin = "gw"
version = 11
cooked_version = 12

df0 = pd.read_csv(
    "~/data/kl/final/thr_{0}_{1}_{2}.csv.gz".format(origin, version, cooked_version),
    compression="gzip",
    index_col=0,
)
origin = "lit"
version = 8

df0lit = pd.read_csv(
    "~/data/kl/final/thr_{0}_{1}_{2}.csv.gz".format(origin, version, cooked_version),
    compression="gzip",
    index_col=0,
)


df = df0.copy()
df_dict = {"gw": df0, "lit": df0lit}

for key, df_ in df_dict.items():
    print(key, df_.shape)

for key, df_ in df_dict.items():
    if key == "lit":
        mask_lit = (df_[up] == 7157) & (df_[dn] == 1026)
        print("filtering out 7157-1026 from lit: {0} rows out ".format(sum(mask_lit)))
        df_dict[key] = df_.loc[~mask_lit].copy()

for key, df_ in df_dict.items():
    print(key, df_.shape)


wdist_grid = 1000
thr_min = 2e-2
thr_max = 4e-1
thr_delta = 2e-3

center_dict = dict()
center_dict["gw"] = (0.21, 0.306)
center_dict["lit"] = (0.16, 0.26)

delta = 0.001
n_grid = 20
half_grid = 0.5 * n_grid


def get_dist(par1, par2, foo, gridn=100, alpha0=0.5, beta0=0.5, verbose=False):
    k, n = par1
    k2, n2 = par2
    delta = 1.0 / gridn
    grid = np.arange(0.0, 1.0, delta)

    pdfs = [beta(alpha0 + k, beta0 + n - k).pdf, beta(alpha0 + k2, beta0 + n2 - k2).pdf]
    dist = foo(*pdfs, grid)
    return dist


def get_thr_study_2tail(df, thrs=None, thr_column=cexp, vote_column=ps):
    hi, lo = thrs
    mask_hi = df[thr_column] > 1.0 - hi
    mask_lo = df[thr_column] < lo
    par_hi = df.loc[mask_hi, vote_column].sum(), sum(mask_hi)
    par_lo = df.loc[mask_lo, vote_column].sum(), sum(mask_lo)
    par_mid = df.loc[~mask_lo & ~mask_hi, vote_column].sum(), sum(~mask_lo & ~mask_hi)
    return par_lo, par_hi, par_mid


data_agg = []
for k, df in df_dict.items():
    hi_center, low_center = center_dict[k]
    a_bound_hi = hi_center - half_grid * delta
    b_bound_hi = hi_center + half_grid * delta + 1e-6
    a_bound_low = low_center - half_grid * delta
    b_bound_low = low_center + half_grid * delta + 1e-6
    hi_grid = np.arange(a_bound_hi, b_bound_hi, delta)
    low_grid = np.arange(a_bound_low, b_bound_low, delta)

    for hi_thr in hi_grid:
        for low_thr in low_grid:
            par_lo, par_hi, par_mid = get_thr_study_2tail(df, (hi_thr, low_thr))
            lo_mid = get_dist(par_lo, par_mid, get_wdist)
            hi_mid = get_dist(par_hi, par_mid, get_wdist)
            lo_hi = get_dist(par_lo, par_hi, get_wdist)
            data_agg += [(k, hi_thr, low_thr, lo_mid, hi_mid, lo_hi)]
            # we want to track the progress
            dfr = pd.DataFrame(
                data_agg,
                columns=[
                    "origin",
                    "hi_thr",
                    "lo_thr",
                    "dist_lomid",
                    "dist_midhi",
                    "dist_hilo",
                ],
            )
            dfr.to_csv(expanduser("~/data/kl/threshold/thr_grid.csv".format(k)))
