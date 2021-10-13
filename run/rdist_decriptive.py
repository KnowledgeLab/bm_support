from datahelpers.constants import (
    iden,
    ye,
    ai,
    ps,
    up,
    dn,
    ar,
    ni,
    cexp,
    qcexp,
    nw,
    wi,
    dist,
    rdist,
    bdist,
    pm,
    cpop,
    cden,
    ct,
    affs,
    aus,
)
from os.path import expanduser, join
import pandas as pd
from bm_support.add_features import generate_feature_groups
from bm_support.add_features import (
    normalize_columns,
    select_feature_families,
    transform_last_stage,
)
from copy import deepcopy
import numpy as np
from numpy.random import RandomState
from bm_support.supervised_aux import find_optimal_model
from bm_support.supervised import logit_pvalue, linear_pvalue
from bm_support.supervised import trim_corrs_by_family
from bm_support.supervised import get_corrs
import seaborn as sns
import json
from bm_support.derive_feature import find_transition, ppf_smart
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor


def sample_one_batch_per_interaction(df, seed=13):
    rns = RandomState(seed)
    dft = df[[up, dn, ye]].drop_duplicates([up, dn, ye])
    dfupdnye = (
        dft.groupby([up, dn])
        .apply(lambda x: x[ye].iloc[rns.randint(x.shape[0])])
        .reset_index()
        .rename(columns={0: ye})
    )
    print(dfupdnye.head())
    dfr = pd.merge(df, dfupdnye, on=[up, dn, ye], how="right")
    return dfr


def aggregate_comm(dfw, comm_col):
    pi2 = dfw.groupby([up, dn, ye, comm_col]).apply(
        lambda x: pd.Series([x[cexp].unique()[0], x["pi_est"].mean()])
    )
    pi3 = pi2.groupby(level=[0, 1, 2]).apply(
        lambda x: pd.Series([x[0].unique()[0], x[1].mean()])
    )
    pi4 = pi3.groupby(level=[0, 1]).apply(
        lambda x: pd.Series([x[0].unique()[0], x[1].mean()])
    )
    return pi4


def compute_pis(df_, y_pred, eps=1e-6):
    # inverse transform
    y_fit_trans = np.exp(y_pred) / (1 + np.exp(y_pred))
    y_fit_trans = (y_fit_trans - eps) / (1.0 - 2 * eps)
    df_["dpred"] = y_fit_trans

    # simple aggregation
    df_["pi_est"] = df_.apply(lambda x: 1 - x["dpred"] if x[ps] else x["dpred"], axis=1)

    pi_comp = df_.groupby([up, dn]).apply(
        lambda x: pd.Series([x[cexp].unique()[0], x["pi_est"].mean()])
    )
    mse_ref = mean_squared_error(pi_comp[0].values, pi_comp[1].values)
    r2_ref = r2_score(pi_comp[0].values, pi_comp[1].values)
    return mse_ref, r2_ref


# parser = argparse.ArgumentParser()
# parser.add_argument('-o', '--origin',
#                     default='lit',
#                     help='type of data to work with [gw, lit]')

# parser.add_argument('-t', '--threshold',
#                     default=2,
#                     type=int,
#                     help='threshold for length of histories')

# parser.add_argument('-f', '--feature-version',
#                     default=15,
#                     help='threshold for length of histories')

# args = parser.parse_args()

origin = "gw"
version = 11
# feat_version = args.feature_version
# len_thr = args.threshold
# origin = args.origin

feat_version = 20
len_thr = 2
# origin = 'lit'
# version = 8
# len_thr = 2

if origin == "lit":
    version = 8
else:
    version = 11

fig_path = expanduser("~/data/kl/figs/tmp/fig_{0}_".format(origin))

ratios = (2.0, 1.0, 1.0)
seed0 = 17
n_trials = 1
datapath = None
verbose = True
model_type = "lr"
cooked_version = 12
# origin = 'litgw'
# version = 1

an_version = 30
excl_columns = ()
target = dist

min_log_alpha = -2
max_log_alpha = 2
log_reg_dict = {"min_log_alpha": min_log_alpha, "max_log_alpha": max_log_alpha}

eps = 0.2
upper_exp, lower_exp = 1 - eps, eps
# thrs = [-1e-8, lower_exp, upper_exp, 1.0001e0]
if datapath:
    col_families = generate_feature_groups(
        expanduser(join(datapath, "v{0}_columns.txt".format(feat_version)))
    )
else:
    col_families = generate_feature_groups(
        expanduser("~/data/kl/columns/v{0}_columns.txt".format(feat_version))
    )

if verbose:
    print(
        "Number of col families: {0}. Keys: {1}".format(
            len(col_families), sorted(col_families.keys())
        )
    )

col_families = {k: v for k, v in col_families.items() if "future" not in k}
if verbose:
    print(
        "Number of col families (excl. future): {0}. Keys: {1}".format(
            len(col_families), sorted(col_families.keys())
        )
    )

columns_interest = [x for sublist in col_families.values() for x in sublist]
if datapath:
    df_path = expanduser(
        join(datapath, "{0}_{1}_{2}.h5".format(origin, version, cooked_version))
    )
else:
    df_path = expanduser(
        "~/data/kl/final/{0}_{1}_{2}.h5".format(origin, version, cooked_version)
    )
df0 = pd.read_hdf(df_path, key="df")


feature_dict = deepcopy(col_families)

families = select_feature_families(an_version)
feature_dict = {k: v for k, v in feature_dict.items() if k in families}
excl_columns = list(set(excl_columns) | {target})

fname = expanduser("~/data/kl/columns/feature_groups.txt")
with open(fname, "r") as f:
    feat_selector = json.load(f)

feature_dict = {
    k: list(v)
    for k, v in feature_dict.items()
    if not any([c in v for c in excl_columns])
}

trial_features = [x for sublist in feature_dict.values() for x in sublist]

feature_dict_inv = {}
for k, v in feature_dict.items():
    feature_dict_inv.update({x: k for x in v})

# (***)

# rns = RandomState(seed0)
# n_trials = 1
# nmax = 10000
# seed = 17
# seeds = rns.randint(nmax, size=n_trials)


cur_features0 = list(set(trial_features) - {"obs_mu"})

case_features = list(set(trial_features) & set(feat_selector["claim"]))
print("len case features ", len(case_features))


# dft = df0.copy()
rd_var = "rdist_abs_trans"
df0["rdist_abs"] = df0["rdist"].abs()

# stretch the target var
eps = 1e-6

df0[rd_var] = df0["rdist_abs"].apply(lambda x: x * (1 - 2 * eps) + eps)
df0[rd_var] = df0[rd_var].apply(lambda x: np.log(x / (1 - x)))

# df_train, df_test = sample_by_length(df0, [up, dn], head=10, seed=10, frac_test=0.3, verbose=True)
# dft = df_train
dft = df0.copy()

# derive change of est. rdist vars
mns = dft.groupby([up, dn, ye]).apply(
    lambda x: pd.Series(x[rdist].mean(), index=["mean_rdist"])
)

long_flag = mns.groupby(level=[0, 1], group_keys=False).apply(lambda x: x.shape[0] > 1)

mns_long = mns.loc[long_flag].reset_index().sort_values([up, dn, ye])
change_flags = (
    mns_long.groupby([up, dn]).apply(lambda x: find_transition(x)).reset_index()
)

dfn = pd.merge(dft, change_flags, on=[up, dn, ye], how="right").sort_values(
    [up, dn, ye]
)

# discard t0 times for each interaction

dfn2_ = dfn[dfn["mean_rdist_prev"].notnull()].copy()
print(dfn2_.shape)

# keep only batches with delta est. rdist non zero

dfn2a = dfn2_.loc[dfn2_["sign_diff_abs"] != 0].copy()


# fill NAs with means

for c in case_features:
    null_mask = dfn2a[c].isnull()
    s = sum(null_mask)
    m = dfn2a.loc[~null_mask, c].mean()
    if s > 0:
        print(c, s, m)
    dfn2a.loc[null_mask, c] = m

# sample : keep only one batch
dfn2 = sample_one_batch_per_interaction(dfn2a, 13)

dfw = transform_last_stage(dfn2, case_features, origin, 0, normalize=True)

cr_abs, cr = get_corrs(dfw, rd_var, case_features, 0.05)
trim_corrs_by_family(cr, feature_dict_inv)


sns_plot = sns.distplot(dfw["rdist_abs"], 20).get_figure()
sns_plot.savefig(fig_path + "rdist_abs.png")
plt.close()

sns_plot = sns.distplot(dfn2[rd_var], 100).get_figure()
sns_plot.savefig(fig_path + "rd_var.png")
plt.close()


# find n_optimal_features

n_optimal_features = 8
y = dfw[rd_var].values
# y = dfw['rdist_abs'].values
x = dfw[case_features].values
ymean = y.mean()
clf, c_opt, acc = find_optimal_model(x, y, n_optimal_features, Lasso, {}, "alpha")
print("c_opt {0}".format(c_opt))
iis = list(np.nonzero(clf.coef_)[0])
nzero_features = [case_features[ii] for ii in iis]

print("*** {0}".format(len(nzero_features)))
print(nzero_features)


lm = Lasso(alpha=c_opt)
x_cut = dfw[nzero_features[:]].values
lm.fit(x_cut, y)
y_fit = lm.predict(x_cut)
r_sk = linear_pvalue(lm, x_cut, y)

df_rsk = pd.DataFrame(
    np.array(r_sk).T,
    columns=["pval", "coeff", "sigma"],
    index=["intercept"] + nzero_features,
).sort_values("pval")

print("p-value ")
print(df_rsk)
print(
    "mse: {0:.3f}, r2 {1:.3f}".format(mean_squared_error(y, y_fit), r2_score(y, y_fit))
)

sns_plot = plt.scatter(y, y_fit, alpha=0.2).get_figure()
sns_plot.savefig(fig_path + "yyfit_lr.png")
plt.close()


mdepth = 6
rf = RandomForestRegressor(max_depth=mdepth)
# rf = RandomForestRegressor(max_depth=None)
x_cut = dfw[nzero_features[:]]
rf.fit(x_cut, y)
y_fit = rf.predict(x_cut)
# r_sk = linear_pvalue(lm, x_cut, yw)
sns_plot = plt.scatter(y, y_fit, alpha=0.2).get_figure()
sns_plot.savefig(fig_path + "yyfit_rf_d{0}.png".format(mdepth))
plt.close()

print(
    "mse: {0:.3f}, r2 {1:.3f}".format(mean_squared_error(y, y_fit), r2_score(y, y_fit))
)


rf = RandomForestRegressor(max_depth=None)
x_cut = dfw[nzero_features[:]]
rf.fit(x_cut, y)
y_fit = rf.predict(x_cut)
# r_sk = linear_pvalue(lm, x_cut, yw)
plt.scatter(y, y_fit, alpha=0.2)

sns_plot = plt.scatter(y, y_fit, alpha=0.2).get_figure()
sns_plot.savefig(fig_path + "yyfit_rf.png")
plt.close()

print(
    "mse: {0:.3f}, r2 {1:.3f}".format(mean_squared_error(y, y_fit), r2_score(y, y_fit))
)

print("rf importances")
print(
    pd.Series(rf.feature_importances_, index=nzero_features).sort_values(
        ascending=False
    )
)


# inverse transform
y_fit_trans = np.exp(y_fit) / (1 + np.exp(y_fit))
y_fit_trans = (y_fit_trans - eps) / (1.0 - 2 * eps)
dfw["dpred"] = y_fit_trans

# simple aggregation
dfw["pi_est"] = dfw.apply(lambda x: 1 - x["dpred"] if x[ps] else x["dpred"], axis=1)

pi_comp = dfw.groupby([up, dn]).apply(
    lambda x: pd.Series([x[cexp].unique()[0], x["pi_est"].mean()])
)
mse_ref = mean_squared_error(pi_comp[0].values, pi_comp[1].values)
r2_ref = r2_score(pi_comp[0].values, pi_comp[1].values)
print("mse: {0:.3f}, r2 {1:.3f}".format(mse_ref, r2_ref))

comm_cols = [c for c in dfw.columns if "commid" in c]

fig = plt.scatter(pi_comp[0], pi_comp[1], alpha=0.2).get_figure()
fig.savefig(fig_path + "interact_yyfit.png")
plt.close()


# for c in comm_cols:
#     dfr = aggregate_comm(dfw, c)
#     y, y_pred = dfr[0].values, dfr[1].values
#     mse = mean_squared_error(y, y_pred)
#     r2 = r2_score(y, y_pred)
#     mse_frac = (mse_ref - mse)/mse
#     r2_frac = - (r2_ref - r2)/r2
#     print('{0} : mse impov {1:.3f}, r2 impov {2:.3f}. mse {3:.3f}, r2 {4:.3f}'.format(c, mse_frac[0], r2_frac, mse, r2))

print("dfw len {0}".format(dfw.shape[0]))
print(
    "accuracy interaction: {0}".format(
        (sum((pi_comp[0] - 0.5) * (pi_comp[1] - 0.5) > 0) / pi_comp.shape[0])
    )
)
