from datahelpers.constants import iden, ye, ai, ps, up, dn, ar, ni, cexp, qcexp, nw, wi, dist, rdist, bdist, pm, \
                                    cpop, cden, ct, affs, aus
from os.path import expanduser, join
import pandas as pd
from bm_support.add_features import generate_feature_groups
from bm_support.add_features import normalize_columns, select_feature_families, transform_last_stage
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
from bm_support.supervised_aux import find_optimal_model, produce_topk_model
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from bm_support.supervised_aux import simple_stratify
from numpy.random import RandomState
from bm_support.supervised_aux import find_optimal_model, produce_topk_model, plot_prec_recall, plot_auc, level_corr
from bm_support.supervised_aux import level_corr_one_tail
from bm_support.supervised import simple_oversample
from bm_support.supervised import logit_pvalue
from statsmodels.discrete.discrete_model import Logit
from bm_support.supervised import get_corrs
from pprint import pprint
from IPython.display import Image
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--origin',
                    default='lit',
                    help='type of data to work with [gw, lit]')

parser.add_argument('-t', '--threshold',
                    default=2,
                    type=int,
                    help='threshold for length of histories')

parser.add_argument('-f', '--feature-version',
                    default=15,
                    help='threshold for length of histories')

args = parser.parse_args()

# origin = 'gw'
# version = 11
# origin = 'lit'
# version = 8
# len_thr = 2

origin = args.origin
if origin == 'lit':
    version = 8
else:
    version = 11

feat_version = args.feature_version
len_thr = args.threshold

ratios = (2., 1., 1.)
seed0 = 17
n_trials = 1
datapath = None
verbose = True
model_type = 'lr'
cooked_version = 12
# origin = 'litgw'
# version = 1

an_version = 20
an_version = 22
excl_columns = ()
target = dist

min_log_alpha = -2
max_log_alpha = 2
log_reg_dict = {'min_log_alpha': min_log_alpha, 'max_log_alpha': max_log_alpha}

eps = 0.2
upper_exp, lower_exp = 1 - eps, eps
# thrs = [-1e-8, lower_exp, upper_exp, 1.0001e0]
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

feature_dict = {k: list(v) for k, v in feature_dict.items() if not any([c in v for c in excl_columns])}

trial_features = [x for sublist in feature_dict.values() for x in sublist]

feature_dict_inv = {}
for k, v in feature_dict.items():
    feature_dict_inv.update({x: k for x in v})

# (***)

rns = RandomState(seed0)
n_trials = 1
nmax = 10000
seed = 17
seeds = rns.randint(nmax, size=n_trials)


cur_features0 = list(set(trial_features) -
                     {'obs_mu'})

df0[cur_features0] = df0[cur_features0].astype(float)
cur_features = list(cur_features0)
remove_features = [1]
reps = []
number_of_features = 5

df_rsk = pd.DataFrame()
how = 'oversample'

while remove_features and len(cur_features) >= number_of_features:
    touch_columns = list(set(cur_features) - {'obs_mu'})
    dfw = transform_last_stage(df0, touch_columns, origin, len_thr, normalize=True, verbose=True)
    dfw = dfw[dfw['obs_mu'].isnull()].copy()
    if how == 'oversample':
        df_train2 = simple_oversample(dfw, bdist, seed=seed, ratios=(1, 1))
    else:
        df_train2 = dfw
    # print('cpoprc: min {0} max {1}'.format(df_train2['cpoprc'].min(), df_train2['cpoprc'].max()))
    # print('### hist of cpoprc')
    # print('{0}'.format(np.histogram(df_train2['cpoprc'])))
    X_train, y_train = df_train2[cur_features], df_train2[bdist]
    print('###: n rows {0}; n features {1}'.format(X_train.shape[0], X_train.shape[1]))
    clf, c_opt, acc = find_optimal_model(X_train, y_train, number_of_features, verbose=True)
    iis = np.nonzero(clf.coef_)[1]
    if iis.size == 0:
        print('!!! all coeffs zero')
        print('{0} {1}'.format(c_opt, acc))
        clf_opt = LogisticRegression('l1', C=1e1*c_opt, solver='liblinear', max_iter=500)
        X_train = df_train2[cur_features]
        clf_opt.fit(X_train, y_train)
        iis = np.nonzero(clf_opt.coef_)[1]
        nzero_features = [cur_features[ii] for ii in iis]
        print('nzero feats')
        print(nzero_features)
        break
    nzero_features = [cur_features[ii] for ii in iis]
    clf_opt = LogisticRegression('l1', C=c_opt, solver='liblinear', max_iter=500)
    X_train = df_train2[nzero_features]

    clf_opt.fit(X_train, y_train)
    r_sk = logit_pvalue(clf_opt, X_train)
    df_rsk = pd.DataFrame(np.array(r_sk)[:, :, 0].T, columns=['pval', 'coeff', 'sigma'],
                          index=['intercept'] + nzero_features).sort_values('pval')
    mask = (df_rsk['pval'] > 0.01) | (df_rsk['coeff'].abs() < 1e-6)
    remove_features = list(set(df_rsk.loc[mask].index) - {'intercept'})
    cur_features = list(set(cur_features) - set(remove_features))
    reps.append((df_rsk, nzero_features, clf_opt, acc))

    base_prec = 1.0 - sum(df_train2[bdist])/df_train2.shape[0]

    print('(***) active features. Prec {0:.4f} prec_base {1:.4f}'.format(acc, base_prec))
    print(df_rsk.loc[~mask])
    print('(***) 2remove features: {0}'.format(remove_features))

accs = [a[-1] for a in reps]
print(accs)
print('***')
print(df_rsk.loc[~mask])

df_rsk.to_csv(expanduser('~/data/kl/features/features_{0}_lenthr_{1}_smethod_{2}_flist_{3}.csv'.format(origin,
                                                                                                len_thr,
                                                                                                how, feat_version)))
