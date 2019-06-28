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
import sys

# selectors = ['claim', 'batch', 'interaction']

# bdist for positive vs negative correlation
# alpha for neutral  vs non neutral; positive correlates with non-neutral
exec_mode = 'bdist'
# exec_mode = 'alpha'

if exec_mode == 'bdist':
    selectors = ['claim', 'batch']
    targets = ['bdist']
elif exec_mode == 'alpha':
    selectors = ['interaction']
    targets = ['alpha']
else:
    sys.exit()


origin = 'gw'
version = 11
cversion = 6

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

if exec_mode == 'bdist':
    fname = expanduser('~/data/kl/columns/feature_groups.txt')
elif exec_mode == 'alpha':
    fname = expanduser('~/data/kl/columns/feature_groups_v2.txt')
else:
    sys.exit()

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

thr_dict = dict()

# thr_dict['gw'] = (0.21, 0.306)
# thr_dict['lit'] = (0.165, 0.26)
thr_dict['gw'] = (0.218, 0.305)
thr_dict['lit'] = (0.157, 0.256)


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
    mask = (dftmp[cexp] <= dn_thr) | (dftmp[cexp] >= 1. - up_thr)
    if exec_mode == 'bdist':
        df_ = dftmp.loc[mask].copy()
        if key == 'lit':
            mask_lit = (df_[up] == 7157) & (df_[dn] == 1026)
            print('filtering out 7157-1026 from lit: {0} rows out '.format(sum(mask_lit)))
            df_ = df_.loc[~mask_lit]

        print('### : below thr: {0}'.format(sum(dftmp[cexp] < dn_thr)))
        print('### : above thr: {0}'.format(sum(dftmp[cexp] > 1. - up_thr)))
        # print(df_[qcexp].value_counts())
        bd_flag = (df_[ps] - df_[cexp] < 0.5).abs()
        df_['bdist'] = bd_flag

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
    elif exec_mode == 'alpha':
        dftmp['alpha'] = mask.astype(int)
        df_dict[key] = dftmp.drop_duplicates([up, dn])
    else:
        sys.exit()


corr_tr = {}
corr_agg = {}

for target in targets:
    print('*** {0}'.format(target))
    corr_tr[target] = {}
    corr_agg[target] = {}
    for k, df in df_dict.items():
        print('*** {0}'.format(k))
        kk = '{0}_{1}'.format(k, target)
        corr_tr[target][k] = {}
        corr_agg[target][k] = {}
        for s in selectors:
            case_features = sorted(set(trial_features) & set(feat_selector[s]) - {target})

            cr_abs, cr = get_corrs(df, target, case_features, -1, dropnas=False, individual_na=True)
            corr_agg[target][k][s] = {'abs': cr_abs, 'nabs': cr}
            print(sum(cr_abs.isnull()), sum(cr.isnull()), cr.shape[0])

            trimmed_cr_df = trim_corrs_by_family(cr, feature_dict_inv)
            corr_tr[target][k][s] = trimmed_cr_df


col_families_inv2 = [{w: k for w in v} for k, v in col_families.items()]
col_families_inv = reduce(lambda a, b: dict(a, **b), col_families_inv2)

example_corr = [v for k, v in corr_agg[targets[0]].items() if 't0' not in k][0]

ordered_inds = {}
for s, item in example_corr.items():
    v = item['nabs']
    vind = list(v.index)
    families = sorted(list(set([col_families_inv[i] for i in vind])))
    good_index = []
    for f in families:
        good_index.extend(sorted([c for c in col_families[f] if c in vind]))
    ordered_inds[s] = good_index

for target in targets:
    cagg = corr_agg[target]
    for mode in selectors:
        print('*** {0}'.format(mode))
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

        ykeys = sorted(cagg.keys())
        ykeys = sorted([c for c in ykeys if 't0' in c]) + sorted([c for c in ykeys if 't0' not in c])

        for k in ykeys:
            cv = cagg[k]
            oin = ordered_inds[mode]
            corr_ = cv[mode]['nabs']
            print(k, sum(corr_.isnull()), corr_.shape, len(oin))

        control = ['abs', 'nabs']

        for ttype in control:
            print('*** {0}'.format(ttype))
            agg_vec = []
            for k in ykeys:
                cr = cagg[k][mode][ttype]
                cols = ordered_inds[mode]
                agg_vec.append(cr.reindex(cols).values)
            ccr = np.vstack(agg_vec)

            dump_file_fname = expanduser('~/data/kl/corrs/corrs_binary_{0}_{1}_{2}_v{3}.csv'.format(target,
                                                                                                    mode,
                                                                                                    ttype, cversion))
            df_corr = pd.DataFrame(ccr.T, index=ordered_ind_flat, columns=ykeys)
            df_corr.to_csv(dump_file_fname)

            ccr_flat = ccr.flatten()
            mask_flat = np.isnan(ccr_flat)
            cmax = abs(ccr_flat[~mask_flat].max())

            print('### corr among feat vecs')
            mask = np.isnan(ccr)
            print(np.corrcoef(ccr[:, ~mask.any(axis=0)]))

            grid_kws = {"height_ratios": (.05, .9), "hspace": .3}
            f, (cbar_ax, ax) = plt.subplots(2, figsize=(36, 6), gridspec_kw=grid_kws)

            mask = np.isnan(ccr)

            ax = sns.heatmap(ccr, mask=mask, ax=ax, cbar_ax=cbar_ax, cmap='RdBu_r',
                             xticklabels=xticks_trans,
                             yticklabels=ykeys,
                             vmin=-cmax, vmax=cmax,
                             cbar_kws={"orientation": "horizontal"})

            ax.set_facecolor('k')
            r = plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                         rotation_mode="anchor")

            fig_fname = expanduser('~/data/kl/figs/corr/heatmap_corr_{0}_{1}_{2}_v{3}.pdf'.format(target,
                                                                                                  mode,
                                                                                                  ttype, cversion))
            plt.savefig(fig_fname, bbox_inches='tight')
            plt.close()
