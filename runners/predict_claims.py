from datahelpers.constants import iden, ye, ai, ps, up, dn, ar, ni, cexp, qcexp, nw, wi, dist, rdist, bdist, pm,                                     cpop, cden, ct, affs, aus
from os.path import expanduser, join
import pandas as pd
from bm_support.add_features import normalize_columns, normalize_columns_with_scaler
from copy import deepcopy
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score,    f1_score, classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from numpy.random import RandomState
from bm_support.supervised_aux import find_optimal_model, produce_topk_model, produce_topk_model_
from bm_support.supervised_aux import plot_prec_recall, plot_auc, plot_auc_pr, level_corr, run_claims
from bm_support.add_features import select_feature_families, transform_last_stage

from bm_support.beta_est import produce_beta_est, produce_claim_valid
from bm_support.beta_est import produce_interaction_plain, estimate_pi
import json
from bm_support.add_features import generate_feature_groups, define_laststage_metrics

from bm_support.math import interpolate_nonuniform_linear, integral_linear, get_function_values, find_bbs

from bm_support.growth import SeqLenGrower
from bm_support.communities import pick_by_community
import pickle
import gzip

# selectors = ['claim', 'batch', 'interaction']

# bdist for positive vs negative correlation
# alpha for neutral  vs non neutral; positive correlates with non-neutral

# exec_mode = 'neutral'
# exec_mode = 'posneg'
exec_mode = 'full'

if exec_mode == 'neutral' or exec_mode == 'posneg':
    selectors = ['interaction']
    targets = ['bint']
elif exec_mode == 'full':
    selectors = ['claim', 'batch']
    targets = ['bdist']

fprefix = f'predict_{exec_mode}'

thr_dict = {'gw': (0.218, 0.305), 'lit': (0.157, 0.256)}

df_dict = {}

for origin in ['gw', 'lit']:
    df_dict[origin] = define_laststage_metrics(origin, predict_mode=exec_mode, verbose=True)
    print(f'>>> {origin} {exec_mode} {df_dict[origin].shape[0]}')


feat_version = 21
an_version = 30

# correlation version

datapath = None
excl_columns = ()

if datapath:
    col_families = generate_feature_groups(expanduser(join(datapath, 'v{0}_columns.txt'.format(feat_version))))
else:
    col_families = generate_feature_groups(expanduser('~/data/kl/columns/v{0}_columns.txt'.format(feat_version)))

fname = expanduser('~/data/kl/columns/feature_groups_v3.txt')
with open(fname, 'r') as f:
    feat_selector = json.load(f)

feature_dict = deepcopy(col_families)

families = select_feature_families(an_version)

feature_dict = {k: list(v) for k, v in feature_dict.items() if not any([c in v for c in excl_columns])}

feature_dict_inv = {}

for k, v in feature_dict.items():
    feature_dict_inv.update({x: k for x in v})


excl_set = set(['bdist_ma_None', 'bdist_ma_2'])

cfeatures0 = list(set(feat_selector['claim']) | set(feat_selector['batch']))
gw_excl = [c for c in cfeatures0 if sum(df_dict['gw'][c].isnull()) > 0]
lit_excl = [c for c in cfeatures0 if sum(df_dict['lit'][c].isnull()) > 0]


cfeatures = (set(feat_selector['claim']) | set(feat_selector['batch'])) - (set(gw_excl) | set(lit_excl))
cfeatures_all = list(cfeatures)
cfeatures_normal = list(cfeatures - excl_set)
cfeatures_excl = list(excl_set)


# cfeatures_container = [cfeatures_normal, cfeatures_excl]
# cfeatures_container = [cfeatures_normal]

verbose = False
seed = 17

n_iter = 1
forest_flag = True
target = 'bdist'
verbose = False
df_package = df_dict
len_thr = 0
oversample = False


# train claims models
rns = RandomState(seed)

container = run_claims(df_package, cfeatures, seed=seed, len_thr=len_thr, forest_flag=forest_flag,
                       n_iter=n_iter, target=target, oversample=oversample,
                       case_features=cfeatures_normal, verbose=False)

fpath = expanduser('~/data/kl/reports/')

with gzip.open(fpath + f'models_claims.pkl.gz', 'wb') as f:
    pickle.dump(container, f)
