from os.path import expanduser, join
from bm_support.add_features import generate_feature_groups, define_laststage_metrics
from bm_support.supervised_aux import run_neut_models
from bm_support.reporting import dump_info
import json


# predict_mode = 'neutral'
predict_mode = 'posneg'
fprefix = f'predict_{predict_mode}'

selectors = ['claim', 'batch']


thr_dict = {'gw': (0.218, 0.305), 'lit': (0.157, 0.256)}

df_dict = {}

for origin in ['gw', 'lit']:
    df_dict[origin] = define_laststage_metrics(origin, predict_mode=predict_mode, verbose=True)
    print(f'>>> {origin} {predict_mode} {df_dict[origin].shape[0]}')


fname = expanduser('~/data/kl/columns/feature_groups_v3.txt')
with open(fname, 'r') as f:
    feat_selector = json.load(f)


fpath = expanduser('~/data/kl/reports/')

model_type = 'rf'

if predict_mode == 'neutral':
    max_len_thr = 11
else:
    max_len_thr = 6

n_iter = 20
fsuffix = 'v5'

# n_iter = 20
# max_len_thr = 6

cfeatures = ['mu*', 'mu*_pct', 'mu*_absmed', 'mu*_absmed_pct',
             # 'degree_source_in', 'degree_source_out',
             # 'degree_target_in', 'degree_target_out'
             'degree_source', 'degree_target'
             ]

extra_features = [c for c in feat_selector['interaction'] if ('same' in c or 'eff' in c) and ('_im_ud' in c)]
cfeatures += extra_features

report, coeffs = run_neut_models(df_dict, cfeatures,
                                 max_len_thr=max_len_thr, n_iter=n_iter,
                                 forest_flag=True, asym_flag=False,
                                 target='bint',
                                 verbose=True)

dump_info(report, coeffs, cfeatures, fsuffix, model_type)

model_type = 'lr'
cfeatures = ['mu*', 'mu*_pct', 'mu*_absmed', 'mu*_absmed_pct',
             # 'degree_source_in', 'degree_source_out',
             # 'degree_target_in', 'degree_target_out'
             'degree_source', 'degree_target'
             ]

extra_features = [c for c in feat_selector['interaction'] if ('same' in c or 'eff' in c) and ('_im_ud' in c)]
cfeatures += extra_features

report, coeffs = run_neut_models(df_dict, cfeatures,
                                 max_len_thr=max_len_thr, n_iter=n_iter,
                                 forest_flag=False, asym_flag=False,
                                 target='bint',
                                 verbose=True)

dump_info(report, coeffs, cfeatures, fsuffix, model_type)

