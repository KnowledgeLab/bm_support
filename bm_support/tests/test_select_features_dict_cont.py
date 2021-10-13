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
from bm_support.supervised_aux import split_three_way
from bm_support.supervised import select_features_dict
from bm_support.add_features import normalize_columns
from bm_support.supervised_aux import study_sample, metric_selector
from functools import partial
from multiprocessing import Pool
from copy import deepcopy
import warnings
from numpy.random import RandomState
import gzip
import pickle
import argparse
from sklearn.model_selection import train_test_split
from bm_support.supervised import simple_stratify
from bm_support.supervised import select_features_dict, logit_pvalue, report_metrics
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")


an_version = 12

# origin = 'litgw'
# version = 1

# origin = 'lit'
# version = 8

origin = "gw"
version = 11

# model_type = 'rf'
# seed0 = 17
n_trials = 1
n_subtrials = 10
n_estimators = 17
# n_estimators = 55
datapath = None
# seed0 = 13
seed0 = 1
n_jobs = 1
verbose = True
model_type = "lr"
model_type = "rf"


min_log_alpha = -1
max_log_alpha = 1
log_reg_dict = {"min_log_alpha": min_log_alpha, "max_log_alpha": max_log_alpha}

eps = 0.2
upper_exp, lower_exp = 1 - eps, eps
# thrs = [-1e-8, lower_exp, upper_exp, 1.0001e0]
if datapath:
    col_families = generate_feature_groups(
        expanduser(join(datapath, "v12_columns.txt"))
    )
else:
    col_families = generate_feature_groups(
        expanduser("~/data/kl/columns/v12_columns.txt")
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
        join(datapath, "{0}_{1}_{2}.h5".format(origin, version, an_version))
    )
else:
    df_path = expanduser(
        "~/data/kl/final/{0}_{1}_{2}.h5".format(origin, version, an_version)
    )

df = pd.read_hdf(df_path, key="df")


# mask: literome - mask out a specific interaction
mask_lit = (df[up] == 7157) & (df[dn] == 1026)

# mask:  interaction with more than 3 claims
thr = 3
mask_len_ = df.groupby(ni).apply(lambda x: x.shape[0]) > thr
mask_max_len = df[ni].isin(mask_len_[mask_len_].index)
print(mask_max_len.shape[0], sum(mask_max_len))

# mask : interactions which are between
eps_window_mean = 0.1
mean_col = 0.5
mask_exp = (
    (df[cexp] <= lower_exp - eps_window_mean)
    | (df[cexp] >= upper_exp + eps_window_mean)
    # | ((df2[cexp] <= (mean_col + eps_window_mean)) & (df2[cexp] >= (mean_col - eps_window_mean)))
)

feature_dict = deepcopy(col_families)
# families = ['affiliations_affind', 'affiliations_comm_size', 'affiliations_ncomms',
#             'affiliations_ncomponents', 'affiliations_size_ulist', 'affiliations_suppind',
#             'ai', 'ar', 'authors_affind', 'authors_comm_size', 'authors_ncomms', 'authors_ncomponents',
#             'authors_size_ulist', 'authors_suppind', 'cden', 'citations',
#             'cite_count', 'cpop', 'delta_year', 'ksst', 'lincscomm_size', 'lincssame_comm',
#             'litgwcsize_dn', 'litgwcsize_up',
#             'litgweff_comm_size', 'litgwsame_comm', 'nhi', 'past_affind',
#             'past_comm_size', 'past_ncomms', 'past_ncomponents',
#             'past_size_ulist', 'past_suppind', 'pre_affs', 'pre_authors', 'time']
families = [
    "affiliations_comm_size",
    "ai",
    "ar",
    "cden",
    "citations",
    "cite_count",
    "cpop",
    "delta_year",
    "ksst",
    "lincscomm_size",
    "lincssame_comm",
    "litgweff_comm_size",
    "litgwsame_comm",
    "nhi",
    "past_affind",
    "past_comm_size",
    "time",
]

feature_dict = {k: v for k, v in feature_dict.items() if k in families}

trial_features = [x for sublist in feature_dict.values() for x in sublist]

feature_dict_inv = {}
for k, v in feature_dict.items():
    feature_dict_inv.update({x: k for x in v})

# mask: not nulls in trial features
masks = []
for c in trial_features:
    masks.append(df[c].notnull())

mask_notnull = masks[0]
for m in masks[1:]:
    mask_notnull &= m

print("Experimental mask len {0}".format(sum(mask_exp)))
print("Number of trial features: {0}".format(len(trial_features)))
print(
    "Number of notnull entries (over all features): {0} from {1}".format(
        sum(mask_notnull), mask_notnull.shape
    )
)

if origin != "gw":
    mask_agg = mask_notnull & ~mask_lit
else:
    mask_agg = mask_notnull

dfw = df.loc[mask_agg].copy()

# metric to optimize for
mm = "accuracy"
# mm = 'precision'

target_column = "litgw_comm_im_undir_wei_pNone_eff_comm_size"

col_pool = [target_column]

feature_dict_new = {
    k: list(v) for k, v in feature_dict.items() if not any([c in v for c in col_pool])
}
len(feature_dict), len(feature_dict_new)

study_sample_flag = False
model_type = "lrg"
model_type = "rfr"

if study_sample_flag:
    r = study_sample(
        dfw=dfw,
        target=dist,
        feature_dict=feature_dict_new,
        metric_mode=mm,
        model_type=model_type,
        n_subtrials=n_subtrials,
        n_estimators=n_estimators,
        log_reg_dict=log_reg_dict,
        verbose=verbose,
        seed=seed0,
    )
else:
    nmax = 5000

    rns = RandomState(seed0)
    seeds = rns.randint(nmax, size=n_trials)

    df_train, df_test, df_valid = split_three_way(dfw, seed0, target_column)

    if model_type == "rf" or model_type == "rfr":
        param_dict = {"n_estimators": n_estimators, "max_features": None, "n_jobs": 1}
    else:
        param_dict = {"n_jobs": 1}

    param_dict["random_state"] = 17

    if model_type == "rf" or model_type == "rfr":
        param_dict = {"n_estimators": n_estimators, "max_features": None, "n_jobs": 1}
    else:
        param_dict = {"n_jobs": 1}

    if model_type == "rf" or model_type == "rfr":
        param_dict["random_state"] = 17
    elif model_type == "lr":
        param_dict["C"] = 1.0

    r = select_features_dict(
        df_train,
        df_test,
        target_column,
        feature_dict,
        model_type="rfr",
        max_features_consider=8,
        metric_mode="accuracy",
        mode_scores=None,
        metric_uniform_exponent=0.5,
        eps_improvement=1e-6,
        model_dict=param_dict,
        verbose=False,
    )

    print(r)
