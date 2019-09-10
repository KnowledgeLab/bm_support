from datahelpers.constants import iden, ye, ai, ps, up, dn, ar, ni, cexp, qcexp, nw, wi, dist, rdist, bdist, pm, \
    cpop, cden, ct, affs, aus
from os.path import expanduser, join
from copy import deepcopy
from numpy.random import RandomState
from bm_support.supervised_aux import run_model, run_claims, run_model_iterate_over_datasets
from bm_support.add_features import select_feature_families, transform_last_stage

import json
from bm_support.add_features import generate_feature_groups, define_laststage_metrics

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
cfeatures_all = list(cfeatures - excl_set)
cfeatures_normal = list(cfeatures - excl_set)
cfeatures_excl = list(excl_set)


verbose = False
seed = 17

n_iter = 20
mode = 'rf'
# mode = 'lr'

target = 'bdist'
verbose = False
df_package = df_dict
len_thr = 0
oversample = False

# train claims models
rns = RandomState(seed)
version = 20

cfeatures = sorted(list(cfeatures - excl_set))

print('***')
print(len(cfeatures))

print('***')
print(cfeatures)
print('***')

if mode == 'rf':
    clf_parameters = {'max_depth': 4, 'n_estimators': 100, 'min_samples_leaf': 20}
else:
    clf_parameters = {'penalty': 'l1', 'solver': 'liblinear', 'max_iter': 100}

extra_parameters = {'min_samples_leaf_frac': 0.05}

container = run_model_iterate_over_datasets(df_package, cfeatures, rns,
                                            target=target, mode=mode, n_splits=3,
                                            clf_parameters=clf_parameters, n_iterations=n_iter)

fpath = expanduser('~/data/kl/reports/')

with gzip.open(fpath + f'models_claims_v{version}.pkl.gz', 'wb') as f:
    pickle.dump(container, f)
