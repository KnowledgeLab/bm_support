from datahelpers.constants import iden, ye, ai, ps, up, dn, ar, ni, cexp, qcexp, nw, wi, dist, rdist, bdist, pm, \
                                    cpop, cden, ct, affs, aus
from os.path import expanduser, join
import pandas as pd
from bm_support.add_features import generate_feature_groups
from bm_support.add_features import normalize_columns, normalize_columns_with_scaler
from bm_support.add_features import select_feature_families, transform_last_stage
from bm_support.supervised_aux import study_sample, metric_selector
from functools import partial
from multiprocessing import Pool
from copy import deepcopy
import warnings
from numpy.random import RandomState
import gzip
import pickle
import argparse
import numpy as np
from numpy.random import RandomState
from bm_support.sampling import sample_by_length
from datahelpers.dftools import keep_longer_histories
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score,\
    f1_score, classification_report, confusion_matrix, balanced_accuracy_score
from itertools import combinations, product
from bm_support.supervised import report_metrics_, train_massif, train_massif_clean, clean_zeros
from bm_support.supervised import invert_bayes, aggregate_over_claims_new, aggregate_over_claims_comm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from bm_support.supervised_aux import simple_stratify
from numpy.random import RandomState
from bm_support.supervised_aux import find_optimal_model, produce_topk_model, produce_topk_model_
from bm_support.supervised_aux import plot_prec_recall, plot_auc, level_corr
from bm_support.supervised_aux import level_corr_one_tail, extract_scalar_metric_from_report
from bm_support.supervised import simple_oversample
from bm_support.supervised import get_corrs
from pprint import pprint
from IPython.display import Image
import seaborn as sns

from bm_support.beta_est import produce_beta_est, produce_claim_valid
from bm_support.beta_est import produce_beta_interaction_est, produce_interaction_est_weighted
from bm_support.beta_est import produce_mu_est, produce_mu_est2, produce_thrs
from bm_support.supervised import trim_corrs_by_family
from bm_support.supervised import get_corrs
import seaborn as sns
import json
from functools import reduce
from bm_support.beta_est import produce_interaction_plain

from bm_support.derive_feature import select_t0, attach_transition_metrics
import matplotlib.pyplot as plt

selectors = ['claim', 'batch']

origin = 'gw'
version = 11

feat_version = 20

if origin == 'lit':
    version = 8
else:
    version = 11

datapath = None
verbose = True
model_type = 'lr'
cooked_version = 12

an_version = 30
excl_columns = ()
target = dist

if datapath:
    col_families = generate_feature_groups(expanduser(join(datapath, 'v{0}_columns.txt'.format(feat_version))))
else:
    col_families = generate_feature_groups(expanduser('~/data/kl/columns/v{0}_columns.txt'.format(feat_version)))

if verbose:
    print('Number of col families: {0}. Keys: {1}'.format(len(col_families), sorted(col_families.keys())))

col_families = {k: v for k, v in col_families.items() if 'future' not in k}
if verbose:
    print('Number of col families (excl. future): {0}. Keys: {1}'.format(len(col_families),
                                                                         sorted(col_families.keys())))

columns_interest = [x for sublist in col_families.values() for x in sublist]
if datapath:
    df_path = expanduser(join(datapath, '{0}_{1}_{2}.h5'.format(origin, version, cooked_version)))
else:
    df_path = expanduser('~/data/kl/final/{0}_{1}_{2}.h5'.format(origin, version, cooked_version))
df0 = pd.read_hdf(df_path, key='df')


feature_dict = deepcopy(col_families)

families = select_feature_families(an_version)
feature_dict = {k: v for k, v in feature_dict.items() if k in families}
excl_columns = list(set(excl_columns) | {target})

fname = expanduser('~/data/kl/columns/feature_groups.txt')
with open(fname, 'r') as f:
    feat_selector = json.load(f)

feature_dict = {k: list(v) for k, v in feature_dict.items() if not any([c in v for c in excl_columns])}

trial_features = [x for sublist in feature_dict.values() for x in sublist]

feature_dict_inv = {}
for k, v in feature_dict.items():
    feature_dict_inv.update({x: k for x in v})

origin = 'lit'
version = 8
cooked_version = 12

an_version = 30
excl_columns = ()
target = dist

min_log_alpha = -2
max_log_alpha = 2
log_reg_dict = {'min_log_alpha': min_log_alpha, 'max_log_alpha': max_log_alpha}

eps = 0.2
upper_exp, lower_exp = 1 - eps, eps

columns_interest = [x for sublist in col_families.values() for x in sublist]
if datapath:
    df_path = expanduser(join(datapath, '{0}_{1}_{2}.h5'.format(origin, version, cooked_version)))
else:
    df_path = expanduser('~/data/kl/final/{0}_{1}_{2}.h5'.format(origin, version, cooked_version))

df0_lit = pd.read_hdf(df_path, key='df')

dft = df0.copy()
dftlit = df0_lit.copy()
df_base = {'gw': dft, 'lit': dftlit}


# define refutation flags:
rd_var = 'rdist_abs_trans'
refute_columns = [c for c in dftlit.columns if 'comm_ave' in c]

thr_dict = {}

thr_dict['gw'] = (0.21, 0.306)
thr_dict['lit'] = (0.16, 0.26)

for k, df_ in df_base.items():
    df_['rdist_abs'] = df_['rdist'].abs()
    for c in refute_columns:
        mask = (df_[ps] == df_[c])
        df_[c] = mask.astype(int)


df_derived = []
df_dict = {}


for key, dftmp in df_base.items():
    print('key {0}'.format(key))
    up_thr, dn_thr = thr_dict[key]
    mask = (dftmp[cexp] < dn_thr) | (dftmp[cexp] > 1. - up_thr)
    df_ = dftmp.loc[mask].copy()
    if key == 'lit':
        mask_lit = (df_[up] == 7157) & (df_[dn] == 1026)
        print('filtering out 7157-1026 from lit: {0} rows out '.format(sum(mask_lit)))
        df_ = df_.loc[~mask_lit]

    print('### : below thr: {0}'.format(sum(dftmp[cexp] < dn_thr)))
    print('### : above thr: {0}'.format(sum(dftmp[cexp] > 1. - up_thr)))
    # print(df_[qcexp].value_counts())
    # bdist correct : 1; incorrect 0.
    bd_flag = (df_[ps] - df_[cexp] < 0.5).abs()
    df_['bdist'] = bd_flag.astype(int)

    df_['bint'] = (df_[cexp] > 0.5).astype(int)

    dft_t0 = select_t0(df_)
    print('t0 df size: {0}'.format(dft_t0.shape[0]))
    for c in refute_columns:
        dft_t0[c] = 0.0
    df_dict[key]['t0'] = dft_t0
    dfn = attach_transition_metrics(df_, 'bdist')

    # dfn = attach_transition_metrics(df_, 'rdist')
    # dfn2 = attach_transition_metrics(dfn, 'rdist_abs')

    dft_gt = select_t0(dfn, t0=False)
    # there is a change in the absolute rdist
    # dfn2_ = dft_gt.loc[dft_gt['sign_diff_abs_bdist'] != 0]
    dfn2_ = dft_gt
    # filter out interesting part
    df_dict[key]['gt'] = dfn2_
    print('gt df size: {0}'.format(dfn2_.shape[0]))


