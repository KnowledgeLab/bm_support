from datahelpers.constants import iden, ye, ai, ps, up, dn, ar, ni, cexp, qcexp, nw, wi, dist, rdist, pm, \
                                    cpop, cden, ct, affs, aus
from os.path import expanduser, join
import pandas as pd
from bm_support.add_features import generate_feature_groups
from bm_support.add_features import normalize_columns, select_feature_families
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

warnings.filterwarnings('ignore')

seed0 = 17
n_trials = 1
datapath = None
verbose = True
model_type = 'rf'
cooked_version = 12
origin = 'litgw'
version = 1
an_version = 20
excl_columns=()
target = dist

min_log_alpha = -2
max_log_alpha = 2
log_reg_dict = {'min_log_alpha': min_log_alpha, 'max_log_alpha': max_log_alpha}

eps = 0.2
upper_exp, lower_exp = 1 - eps, eps
# thrs = [-1e-8, lower_exp, upper_exp, 1.0001e0]
if datapath:
    col_families = generate_feature_groups(expanduser(join(datapath, 'v13_columns.txt')))
else:
    col_families = generate_feature_groups(expanduser('~/data/kl/columns/v13_columns.txt'))

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
df = pd.read_hdf(df_path, key='df')

# mask: literome - mask out a specific interaction
mask_lit = (df[up] == 7157) & (df[dn] == 1026)

# mask : interactions which are between
eps_window_mean = 0.1
mean_col = 0.5

mask_exp = ((df[cexp] <= lower_exp - eps_window_mean) | (df[cexp] >= upper_exp + eps_window_mean)
            # | ((df2[cexp] <= (mean_col + eps_window_mean)) & (df2[cexp] >= (mean_col - eps_window_mean)))
            )

feature_dict = deepcopy(col_families)

families = select_feature_families(an_version)
feature_dict = {k: v for k, v in feature_dict.items() if k in families}
excl_columns = list(set(excl_columns) | {target})

feature_dict = {k: list(v) for k, v in feature_dict.items() if not any([c in v for c in excl_columns])}

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

print('Experimental mask len {0}'.format(sum(mask_exp)))
print('Number of trial features: {0}'.format(len(trial_features)))
print('Number of notnull entries (over all features): {0} from {1}'.format(sum(mask_notnull), mask_notnull.shape))

if origin != 'gw':
    mask_agg = mask_notnull & ~mask_lit
else:
    mask_agg = mask_notnull

dfw = df.loc[mask_agg].copy()

thr = 2
mask_len_ = (dfw.groupby([up, dn]).apply(lambda x: x.shape[0]) > thr)
updns = mask_len_[mask_len_].reset_index()[[up, dn]]
dfw = dfw.merge(updns, how='right', on=[up, dn])

if model_type == 'lr' or model_type == 'lrg':
    dfw = normalize_columns(dfw, trial_features)

df_train, df_test = sample_by_length(dfw, (up, dn), head=10, seed=11, frac_test=0.4, verbose=True)

n_trees = 57
n_massiv = 30
thr = 15
rr, massif = train_massif_clean(df_train, trial_features, dist, 11, n_massiv, n_trees)
df_obj2 = invert_bayes(df_test, massif, trial_features, True)
df_obj2 = keep_longer_histories(df_obj2, thr, [up, dn])
y_test, y_pred = aggregate_over_claims_new(df_obj2)
print(len(y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)


probs = [rfm.predict_proba(df_test[trial_features])[np.newaxis, ...] for rfm in massif]
arr_probs = np.concatenate(probs, axis=0)
p_min = 1e-3
arr_probs2 = clean_zeros(arr_probs, p_min, 2)
arr_probs2 = np.sum(arr_probs, axis=0)
arr_probs2 = arr_probs2 / np.sum(arr_probs2, axis=1)[..., np.newaxis]
c_pred = np.argmax(arr_probs2, axis=1)
c_test = 2*df_test[ps]
print(len(c_pred), len(c_test))
conf_matrix = confusion_matrix(c_test, c_pred)
print(conf_matrix)