from datahelpers.constants import iden, ye, ai, ps, up, dn, ar, ni, cexp, qcexp, nw, wi, dist
from os.path import expanduser, join
import pandas as pd
from bm_support.add_features import generate_feature_groups
from bm_support.add_features import select_feature_families, transform_last_stage
from copy import deepcopy
import json
from bm_support.supervised_aux import run_neut_models
from bm_support.reporting import dump_info

predict_mode = 'neutral'
predict_mode = 'posneg'
fprefix = f'predict_{predict_mode}'
fsuffix = 'v0'

thr_dict = {'gw': (0.218, 0.305), 'lit': (0.157, 0.256)}

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

columns_interest = [x for sublist in col_families.values() for x in sublist]
if datapath:
    df_path = expanduser(join(datapath, '{0}_{1}_{2}.h5'.format(origin, version, cooked_version)))
else:
    df_path = expanduser('~/data/kl/final/{0}_{1}_{2}.h5'.format(origin, version, cooked_version))

df0_lit = pd.read_hdf(df_path, key='df')


df_base = {'gw': df0, 'lit': df0_lit}

# define refutation flags:
rd_var = 'rdist_abs_trans'
refute_columns = [c for c in df_base['lit'].columns if 'comm_ave' in c]

for k, df_ in df_base.items():
    df_['rdist_abs'] = df_['rdist'].abs()
    for c in refute_columns:
        mask = (df_[ps] == df_[c])
        df_[c] = mask.astype(int)
"""
df_dict = {}
for key, dftmp in df_base.items():
    print('key {0}'.format(key))
    up_thr, dn_thr = thr_dict[key]
    # mask the 
    mask = (dftmp[cexp] < dn_thr) | (dftmp[cexp] > 1. - up_thr)
    df_ = dftmp.loc[mask].copy()
    if key == 'lit':
        mask_lit = (df_[up] == 7157) & (df_[dn] == 1026)
        print('filtering out 7157-1026 from lit: {0} rows out '.format(sum(mask_lit)))
        df_ = df_.loc[~mask_lit]

    print('### : below thr: {0}'.format(sum(dftmp[cexp] < dn_thr)))
    print('### : above thr: {0}'.format(sum(dftmp[cexp] > 1. - up_thr)))
    bd_flag = (df_[ps] - df_[cexp] < 0.5).abs()
    df_['bdist'] = bd_flag.astype(int)

    df_['bint'] = (df_[cexp] > 0.5).astype(int)

    dft_t0 = select_t0(df_)
    print('t0 df size: {0}'.format(dft_t0.shape[0]))
    for c in refute_columns:
        dft_t0[c] = 0.0
    df_dict[key + '_t0'] = dft_t0
    dfn = attach_transition_metrics(df_, 'bdist')

    # dfn = attach_transition_metrics(df_, 'rdist')
    # dfn2 = attach_transition_metrics(dfn, 'rdist_abs')

    dft_gt = select_t0(dfn, t0=False)
    # there is a change in the absolute rdist
    # dfn2_ = dft_gt.loc[dft_gt['sign_diff_abs_bdist'] != 0]
    dfn2_ = dft_gt
    # filter out interesting part
    df_dict[key + '_gtdiff'] = dfn2_
    print('gt df size: {0}'.format(dfn2_.shape[0]))
"""


# define k, n for interactions -> save to df_qm
df_knq = {}
uniq_kn = set()
for k, df_ in df_base.items():
    dft = df_.groupby([up, dn]).apply(lambda x: pd.Series([sum(x[ps]), x.shape[0], x[cexp].iloc[0]],
                                                          index=['k', 'n', 'q']))
    arr = dft[['k', 'n']].apply(lambda x: tuple(x), axis=1)
    uniq_kn |= set(arr.unique())
    df_knq[k] = dft


# define year ymin, ymax for interactions -> df_years
df_years = {}
for k, df_ in df_base.items():
    dft = df_.groupby([up, dn]).apply(lambda x: pd.Series([x[ye].min(), x[ye].max()],
                                                          index=['ymin', 'ymax']))
    df_years[k] = dft.reset_index()

dfye = pd.concat(df_years.values())
dfye = dfye.groupby([up, dn]).apply(lambda x: pd.Series([x.ymin.min(), x.ymax.max()],
                                                        index=['ymin', 'ymax']))


# load degrees per interaction
df_degs = pd.read_csv(expanduser('~/data/kl/comms/interaction_network/updn_degrees.csv.gz'), index_col=0)
# load beta distribution distances per k, n
df_dist = pd.read_csv('~/data/kl/qmu_study/uniq_kn_dist.csv', index_col=0)

# aggregate beta distr dist, degree data with the interaction data frame
df_interaction = {}
for k, df_ in df_knq.items():
    dfr = pd.merge(df_.reset_index(), df_dist, on=['k', 'n'])
    dfr = dfr.loc[dfr.n > 0]
    dfr = pd.merge(dfr, df_degs, on=[up, dn])

    dfr['r'] = dfr.k/dfr.n
    dfr['qabs'] = (dfr.q - 0.5).abs()
    dfr['mu*'] = 1 - dfr['dist']
    dfr['pct_mu*'] = dfr['mu*'].rank(pct=True)
    dfr['abs_pct_mu*'] = (dfr['pct_mu*'] - dfr['pct_mu*'].mean()).abs()
    corr_pct = dfr[['q', 'pct_mu*']].corr().values[0, 1]
    corr_abs_pct = dfr[['qabs', 'abs_pct_mu*']].corr().values[0, 1]

    thr_up, thr_dn = thr_dict[k]
    mask_pos = (dfr.q > (1. - thr_up))
    mask_neg = (dfr.q < thr_dn)

    dfr['bdist'] = 0.
    if predict_mode == 'neutral':
        dfr.loc[mask_neg | mask_pos, 'bdist'] = 1.
    else:
        dfr.loc[mask_neg, 'bdist'] = 1.
        dfr = dfr.loc[mask_neg | mask_pos]

    dfr['muabs'] = (dfr['mu*'] - dfr['mu*'].mean()).abs()

    if k == 'lit':
        mask_lit = (dfr[up] == 7157) & (dfr[dn] == 1026)
        print('filtering out 7157-1026 from lit: {0} rows out '.format(sum(mask_lit)))
        dfr = dfr.loc[~mask_lit]
    df_interaction[k] = dfr.copy()

fpath = expanduser('~/data/kl/reports/')

model_type = 'rf'
cfeatures = ['pct_mu*',
             # 'abs_pct_mu*',
             'updeg_st', 'dndeg_st', 'effdeg_st',
             'updeg_end', 'dndeg_end', 'effdeg_end'
             ]

if predict_mode == 'neutral':
    max_len_thr = 21
else:
    max_len_thr = 11

n_iter = 20

report, coeffs = run_neut_models(df_interaction, cfeatures,
                                 max_len_thr=max_len_thr, n_iter=n_iter,
                                 forest_flag=True, asym_flag=False,
                                 verbose=True)

dump_info(report, coeffs, cfeatures, fsuffix, model_type)

model_type = 'lr'
cfeatures = ['pct_mu*',
             'abs_pct_mu*',
             'updeg_st', 'dndeg_st', 'effdeg_st',
             'updeg_end', 'dndeg_end', 'effdeg_end']

report, coeffs = run_neut_models(df_interaction, cfeatures,
                                 max_len_thr=max_len_thr, n_iter=n_iter,
                                 forest_flag=False, asym_flag=False,
                                 verbose=True)

dump_info(report, coeffs, cfeatures, fsuffix, model_type)

