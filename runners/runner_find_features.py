from datahelpers.constants import iden, ye, ai, ps, up, dn, ar, ni, cexp, qcexp, nw, wi, dist, rdist, pm, \
                                    cpop, cden, ct, affs, aus
import numpy as np
from sklearn.model_selection import train_test_split
from os.path import expanduser
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns
from bm_support.supervised import simple_stratify, train_forest, train_massif
import pandas as pd
from bm_support.add_features import train_test_split_key, generate_feature_groups

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from bm_support.supervised import select_features_dict
from copy import deepcopy
import warnings
from numpy.random import RandomState
from bm_support.supervised import report_metrics

warnings.filterwarnings('ignore')

fpath = expanduser('~/data/kl/figs/rf')

an_version = 12
origin = 'lit'
version = 8
eps = 0.2
upper_exp, lower_exp = 1 - eps, eps
thrs = [-1e-8, lower_exp, upper_exp, 1.0001e0]
metric_selector = dict(zip(['corr',  'accuracy', 'precision', 'recall', 'f1'], range(5)))

# cur_metric_columns = ['cpoprc', 'cdenrc', 'ksstrc']

col_families = generate_feature_groups(expanduser('~/data/kl/columns/v12_columns.txt'))
columns_interest = [x for sublist in col_families.values() for x in sublist]


df = pd.read_hdf('/Users/belikov/data/kl/final/{0}_{1}_{2}.h5'.format(origin, version, an_version, ), key='df')

# mask: literome - mask out a specific interaction
mask_lit = (df[up] == 7157) & (df[dn] == 1026)

# mask:  interaction with more than 3 claims
thr = 3
mask_len_ = (df.groupby(ni).apply(lambda x: x.shape[0]) > thr)
mask_max_len = df[ni].isin(mask_len_[mask_len_].index)
print(mask_max_len.shape[0], sum(mask_max_len))

# mask : interactions which are between
eps_window_mean = 0.1
mean_col = 0.5
mask_exp = ((df[cexp] <= lower_exp - eps_window_mean) | (df[cexp] >= upper_exp + eps_window_mean)
            # | ((df2[cexp] <= (mean_col + eps_window_mean)) & (df2[cexp] >= (mean_col - eps_window_mean)))
            )

fdict = {}
fdict['cpop'] = list(col_families['cpop'])
fdict['future_comm_size'] = list(col_families['future_comm_size'])
fdict = deepcopy(col_families)
trial_features = [x for sublist in fdict.values() for x in sublist]

# mask: not nulls in trial features
masks = []
for c in trial_features:
    masks.append(df[c].notnull())

mask_notnull = masks[0]
for m in masks[1:]:
    mask_notnull &= m

print(sum(mask_exp))
print(len(trial_features))
print(sum(mask_notnull), mask_notnull.shape)

mask_agg = mask_notnull & mask_lit
dfw = df.loc[mask_notnull].copy()

seed = 17
df_train, df_testgen = train_test_split(dfw, test_size=0.4,
                                        random_state=seed, stratify=dfw[dist])
df_valid, df_test = train_test_split(df_testgen, test_size=0.5,
                                     random_state=seed)
df_train2 = simple_stratify(df_train, dist, seed, ratios=(2, 1, 1))


rns = RandomState(seed=11)
seeds = rns.randint(50, size=5)
mm = 'precision'

if model_type == 'rf':
    param_dict = {'n_estimators': 137, 'random_state': 13, 'max_features':None}
elif model_type == 'lr':
    param_dict = {'C': 10}

meta_agg = []
print(seeds)
for seed in seeds:
    param_dict['random_state'] = seed
    df_train2 = simple_stratify(df_train, dist, seed, ratios=(2, 1, 1))
    cfeatures, cmetrics, cvector_metrics, model_rf = select_features_dict(df_train2, df_test, dist, fdict,
                                                                          model_type='rf',
                                                                          max_features_consider=8,
                                                                          metric_mode=mm,
                                                                          model_dict=param_dict,
                                                                          verbose=True)
    vscalar, vvector = report_metrics(model_rf, df_valid[cfeatures], df_valid[dist])
    meta_agg.append((seed, cfeatures, cmetrics, cvector_metrics, vscalar, vvector, model_rf))

# print([x[4][metric_selector[mm]] for x in meta_agg])
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_ylabel(mm)
for item in meta_agg:
    seed, cfeatures, cmetrics, cvector_metrics, vscalar, vvector, model_rf = item
    xcoords = list(range(len(cmetrics)))
    ax.plot(xcoords, np.array(cmetrics)[:, metric_selector[mm]])