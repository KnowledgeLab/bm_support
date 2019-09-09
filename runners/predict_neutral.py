from os.path import expanduser, join
from bm_support.add_features import generate_feature_groups, define_laststage_metrics
from bm_support.supervised_aux import run_neut_models, run_model_iterate_over_datasets
from bm_support.reporting import dump_info
from numpy.random import RandomState
import json


# predict_mode = 'neutral'
predict_mode = 'posneg'
fprefix = f'predict_{predict_mode}'

selectors = ['claim', 'batch']

# model_type = 'lr'
model_type = 'rf'

if model_type == 'rf':
    forest_flag = True
else:
    forest_flag = False

thr_dict = {'gw': (0.218, 0.305), 'lit': (0.157, 0.256)}

df_dict = {}

for origin in ['gw', 'lit']:
    df_dict[origin] = define_laststage_metrics(origin, predict_mode=predict_mode, verbose=True)
    print(f'>>> {origin} {predict_mode} {df_dict[origin].shape[0]}')


fname = expanduser('~/data/kl/columns/feature_groups_v3.txt')
with open(fname, 'r') as f:
    feat_selector = json.load(f)


fpath = expanduser('~/data/kl/reports/')

if predict_mode == 'neutral':
    max_len_thr = 11
else:
    max_len_thr = 6

max_len_thr = 1
n_iter = 20
fsuffix = 'v5'
mode = 'rf'
seed = 17
rns = RandomState(seed)
target='bint'


fprefix = f'predict_{predict_mode}_nodeg'

cfeatures = ['mu*', 'mu*_pct', 'mu*_absmed', 'mu*_absmed_pct',
             # 'degree_source_in', 'degree_source_out',
             # 'degree_target_in', 'degree_target_out'
             # 'degree_source', 'degree_target'
             ]

extra_features = [c for c in feat_selector['interaction'] if ('same' in c or 'eff' in c) and ('_im_ud' in c)]
cfeatures += extra_features

if mode == 'rf':
    clf_parameters = {'max_depth': 2, 'n_estimators': 100, 'min_samples_leaf': 20}
else:
    clf_parameters = {'penalty': 'l1', 'solver': 'liblinear', 'max_iter': 100}

extra_parameters = {'min_samples_leaf_frac': 0.05}

container = run_model_iterate_over_datasets(df_dict, cfeatures,
                                            rns, target=target, mode=mode, n_splits=3,
                                            clf_parameters=clf_parameters, n_iterations=n_iter)


dump_info(container, fsuffix, model_type, fprefix=fprefix)


