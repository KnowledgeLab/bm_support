from datahelpers.constants import iden, ye, ai, ps, up, dn, ar, ni, cexp, qcexp, nw, wi, dist, rdist, bdist, pm, \
                                    cpop, cden, ct, affs, aus
from os.path import expanduser, join
import pandas as pd
from bm_support.add_features import generate_feature_groups
from bm_support.add_features import normalize_columns, select_feature_families, transform_last_stage
from copy import deepcopy
import numpy as np
from bm_support.supervised import trim_corrs_by_family
from bm_support.supervised import get_corrs
import seaborn as sns
import json
from functools import reduce
from bm_support.derive_feature import select_t0, attach_transition_metrics
import matplotlib.pyplot as plt

selectors = ['claim', 'batch', 'interaction']
mode = 'batch'

origin = 'gw'
version = 11

feat_version = 20
len_thr = 2

if origin == 'lit':
    version = 8
else:
    version = 11

ratios = (2., 1., 1.)
seed0 = 17
n_trials = 1
datapath = None
verbose = True
model_type = 'lr'
cooked_version = 12

an_version = 30
excl_columns = ()
target = dist

min_log_alpha = -2
max_log_alpha = 2
log_reg_dict = {'min_log_alpha': min_log_alpha, 'max_log_alpha': max_log_alpha}

eps = 0.2
upper_exp, lower_exp = 1 - eps, eps
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

for k, df_ in df_base.items():
    df_['rdist_abs'] = df_['rdist'].abs()
    for c in refute_columns:
        mask = (df_[ps] == df_[c])
        df_[c] = mask.astype(int)


df_derived = []
df_dict = {}
for key, df_ in df_base.items():
    dft_t0 = select_t0(df_)
    for c in refute_columns:
        dft_t0[c] = 0.0
    df_dict[key + '_t0'] = dft_t0
    dfn = attach_transition_metrics(df_, 'rdist')
    dfn2 = attach_transition_metrics(dfn, 'rdist_abs')

    dft_gt = select_t0(dfn2, t0=False)
    # there is a change in the absolute rdist
    dfn2_ = dft_gt.loc[dft_gt['sign_diff_abs_rdist_abs'] != 0]
    # filter out interesting part
    df_dict[key + '_gtdiff'] = dfn2_

corr_tr = {}
corr_agg = {}

targets = ['rdist_abs', 'bdist']

for target in targets:
    for k, df in df_dict.items():
        kk = '{0}_{1}'.format(k, target)
        corr_tr[k] = {}
        corr_agg[k] = {}
        for s in selectors:
            case_features = sorted(set(trial_features) & set(feat_selector[s]))

            cr_abs, cr = get_corrs(df, 'rdist_abs', case_features, -1, dropnas=False, individual_na=True)
            corr_agg[k][s] = {'abs': cr_abs, 'nabs': cr}
            print(sum(cr_abs.isnull()), sum(cr.isnull()), cr.shape[0])

            trimmed_cr_df = trim_corrs_by_family(cr, feature_dict_inv)
            corr_tr[k][s] = trimmed_cr_df


col_families_inv2 = [{w: k for w in v} for k, v in col_families.items()]
col_families_inv = reduce(lambda a, b: dict(a, **b), col_families_inv2)

example_corr = [corr_agg[k] for k in corr_agg.keys() if 't0' not in k][0]

ordered_inds = {}
for s, item in example_corr.items():
    v = item['nabs']
    vind = list(v.index)
    families = sorted(list(set([col_families_inv[i] for i in vind])))
    good_index = []
    for f in families:
        good_index.extend(sorted([c for c in col_families[f] if c in vind]))
    ordered_inds[s] = good_index

for mode in selectors:
    ordered_ind_flat = ordered_inds[mode]

    xticks = ordered_ind_flat
    xticks = [col_families_inv[x] for x in xticks]
    xticks_trans = []
    xticks_appearance = []
    for x in xticks:
        if x in xticks_appearance:
            xticks_trans.append('')
        else:
            xticks_trans.append(x)
            xticks_appearance.append(x)

    ykeys = sorted(corr_agg.keys())
    ykeys = sorted([c for c in ykeys if 't0' in c]) + sorted([c for c in ykeys if 't0' not in c])

    for k in ykeys:
        cv = corr_agg[k]
        oin = ordered_inds[mode]
        corr_ = cv[mode]['nabs']
        print(k, sum(corr_.isnull()), corr_.shape, len(oin))

    control = ['abs', 'nabs']

    for ttype in control:
        agg_vec = []
        for k in ykeys:
            cr = corr_agg[k][mode][ttype]
            cols = ordered_inds[mode]
            agg_vec.append(cr.reindex(cols).values)
        ccr = np.vstack(agg_vec)

        grid_kws = {"height_ratios": (.05, .9), "hspace": .3}
        f, (cbar_ax, ax) = plt.subplots(2, figsize=(36, 6), gridspec_kw=grid_kws)

        mask = np.isnan(ccr)

        ax = sns.heatmap(ccr, mask=mask, ax=ax, cbar_ax=cbar_ax, cmap='RdBu_r',
                         xticklabels=xticks_trans,
                         yticklabels=ykeys,
                         cbar_kws={"orientation": "horizontal"})

        ax.set_facecolor('k')
        r = plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor")
        plt.savefig(expanduser('~/data/kl/figs/corr/heatmap_corr_{0}_{1}_v4.pdf'.format(mode, ttype)),
                    bbox_inches='tight')
        plt.close()

