from os.path import expanduser, join
import pandas as pd
from bm_support.add_features import normalize_columns, select_feature_families, transform_last_stage
from copy import deepcopy
import numpy as np
from bm_support.supervised import trim_corrs_by_family
from bm_support.supervised import get_corrs
import seaborn as sns
import json
from functools import reduce
import matplotlib.pyplot as plt
import sys
from bm_support.add_features import generate_feature_groups, define_laststage_metrics


replace_dict = {
    # interaction partition size
    'litgw_comm_dyneffcommsize' :'IPS',
    # interaction partition position
    'litgw_comm_dynsamecomm' : 'IPP',
    # mean claim percentile
    'mu*_pct': 'MCP',
    # absolute median claim percentile
    'mu*_absmed_pct': 'AMMCP',

    'rncomms': 'CCN',

    'suppind': 'SIND',

    'affind': 'AIND',

    'nhi': 'NHI',

    'pre': 'NW',

    'rcomm_size': 'CAS',

    'rrelcomm_size': 'CRS',

    'cden': 'CD',

    'cpop': 'CP',

    'ksst': 'FLAT',
}


def push_subset_right(flist, subset):
    subset = sorted(subset)
    for s in subset:
        if s in flist:
            j = flist.index(s)
            flist.pop(j)
            flist += [s]
    return flist


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
else:
    sys.exit()

fprefix = f'predict_{exec_mode}'

thr_dict = {'gw': (0.218, 0.305), 'lit': (0.157, 0.256)}

df_dict = {}

for origin in ['gw', 'lit']:
    df_dict[origin] = define_laststage_metrics(origin, predict_mode=exec_mode, verbose=True)
    print(f'>>> {origin} {exec_mode} {df_dict[origin].shape[0]}')


feat_version = 21
an_version = 30

# correlation version
cversion = 8

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

corr_tr = {}
corr_agg = {}

excl_symbol_list = ['1', '3', 'csize_dn', 'csize_up', 'bdist']

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
            case_features = sorted(set(feat_selector[s]) - {target})
            case_features = [c for c in case_features if not any([x in c for x in excl_symbol_list])]
            if exec_mode == 'neutral' or  exec_mode == 'posneg':
                extra_feautres = ['mu*_pct', 'mu*_absmed_pct']
            else:
                extra_feautres = []
            case_features += extra_feautres
            # feature_dict_inv.update(dict(zip(extra_feautres, extra_feautres)))
            cr_abs, cr = get_corrs(df, target, case_features, -1, dropnas=False, individual_na=True)
            cr = -cr
            corr_agg[target][k][s] = {'abs': cr_abs, 'nabs': cr}
            print(sum(cr_abs.isnull()), sum(cr.isnull()), cr.shape[0])
            if s == 'interaction':
                trimmed_cr_df = cr
            else:
                trimmed_cr_df = trim_corrs_by_family(cr, feature_dict_inv)
            corr_tr[target][k][s] = trimmed_cr_df


col_families_inv2 = [{w: k for w in v} for k, v in col_families.items()]
col_families_inv = reduce(lambda a, b: dict(a, **b), col_families_inv2)

example_corr = [v for k, v in corr_agg[targets[0]].items() if 't0' not in k][0]
ordered_inds = {}

for s, item in example_corr.items():
    v = item['nabs']
    vind = list(v.index)
    if s != 'interaction':
        families = sorted(list(set([col_families_inv[i] for i in vind])))
        good_index = []
        print(s)
        for f in families:
            good_index.extend(sorted([c for c in col_families[f] if c in vind]))
    else:
        good_index = vind
    if s == 'claim':
        # recast rcommrel to rrelcom
        good_index_tr = [x.replace('rcommrel', 'rrelcomm')
                         for x in good_index]
        aux = [x.split('_') for x in good_index_tr]
        aux = [x[1:3] + x[0:1] + x[3:]
               if any(['affind' in y for y in x]) or
               any(['rcomm' in y for y in x]) or
               any(['rrelcomm' in y for y in x]) or
               any(['count' in y for y in x]) else x
               for x in aux]
        aux = list(zip(good_index, aux))
        good_index_ = sorted(aux, key=lambda x: x[1])
        good_index = [x for x, y in good_index_]
        good_index = push_subset_right(good_index, feature_dict['citations'])
        rename_dict = {v: '_'.join([x for x in k if x]) for v, k in good_index_}
        ordered_inds[s] = good_index
    elif s == 'batch' or s == 'interaction':
        # exclude non dynamic communities
        good_index = [x
               for x in good_index
               if (not 'litgw_comm' in x) or
               ('litgw_comm' in x and 'dyn' in x)]
        aux = [x.split('_') for x in good_index]
        aux = [x[:2] + [''.join(x[6:])] + x[2:6]
                    for x in aux]
        aux = [x[1:2] + x[0:1] + x[2:]
               if any(['rncomms' in y for y in x]) or
               any(['suppind' in y for y in x]) else x
               for x in aux]
        aux = list(zip(good_index, aux))
        good_index_ = sorted(aux, key=lambda x: x[1])
        rename_dict = {v: '_'.join([x for x in k if x]) for v, k in good_index_}
        good_index = [x for x, y in good_index_]
    ordered_inds[s] = good_index, rename_dict

for target in targets[:]:
    cagg = corr_agg[target]
    for mode in selectors[:]:
        print('*** {0}'.format(mode))
        ordered_ind_flat, rn = ordered_inds[mode]

        xticks = ordered_ind_flat
        if mode != 'interaction':
            xticks = [col_families_inv[x] for x in xticks]
            xticks_trans = []
            xticks_appearance = []
            for x in xticks:
                if x in xticks_appearance:
                    xticks_trans.append('')
                else:
                    xticks_trans.append(x)
                    xticks_appearance.append(x)
        else:
            xticks_trans = []
            xticks_appearance = []

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
                agg_vec.append(cr.reindex(ordered_ind_flat).values)
            ccr = np.vstack(agg_vec)

            dump_file_fname = expanduser(f'~/data/kl/corrs/corrs_binary_{target}_{exec_mode}_'
                                         f'{mode}_{ttype}_v{cversion}.csv')
            df_corr = pd.DataFrame(ccr.T, index=ordered_ind_flat, columns=ykeys)
            df_corr.to_csv(dump_file_fname)
            ccr = df_corr.values.T

            ccr_flat = ccr.flatten()
            mask_flat = np.isnan(ccr_flat)
            cmax = abs(ccr_flat[~mask_flat].max())

            print('### corr among feat vecs')
            mask = np.isnan(ccr)
            print(np.corrcoef(ccr[:, ~mask.any(axis=0)]))

            grid_kws = {"height_ratios": (.05, .9), "hspace": .3}
            f, (cbar_ax, ax) = plt.subplots(2, figsize=(36, 6), gridspec_kw=grid_kws)

            mask = np.isnan(ccr)

            # cmap = sns.diverging_palette(240, 10, l=5, n=90, as_cmap=True)
            # cmap = sns.color_palette("RdBu_r", 90)
            cmap = "RdBu_r"

            rename_dict2 = {}
            for v in rn.values():
                vv = str(v)
                for qin, qout in replace_dict.items():
                    vv = vv.replace(qin, qout)
                rename_dict2[v] = vv

            xticks = [rename_dict2[rn[k]] for k in ordered_ind_flat]

            ax = sns.heatmap(ccr, mask=mask, ax=ax, cbar_ax=cbar_ax, cmap=cmap,
                             xticklabels=xticks,
                             # xticklabels=xticks_trans,
                             yticklabels=ykeys,
                             vmin=-cmax, vmax=cmax,
                             cbar_kws={"orientation": "horizontal"})

            ax.set_facecolor('k')
            r = plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                         rotation_mode="anchor")

            fig_fname = expanduser(f'~/data/kl/figs/corr/heatmap_corr_{target}_{exec_mode}_'
                                   f'{mode}_{ttype}_v{cversion}.pdf')
            plt.savefig(fig_fname, bbox_inches='tight')
            plt.close()
