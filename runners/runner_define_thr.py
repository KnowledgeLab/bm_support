import pandas as pd
import matplotlib.pyplot as plt
from math import isinf, isnan
from os.path import expanduser
import numpy as np
from cvxpy import Variable, Parameter, Minimize, Problem
from cvxpy import multiply as cvx_mul
from cvxpy import sum as cvx_sum
from scipy.stats import beta, entropy, kstat
from scipy.stats import uniform
import seaborn

up, dn, ps, pm, cexp = 'up', 'dn', 'pos', 'pmid', 'cdf_exp'


def yield_kl_dist(a, n, grid, precomp_dict=None, pa=1, pb=1):
    b = n - a
    if (a, b) in precomp_dict.keys():
        ent = precomp_dict[(a, b)]
    else:
        pk = beta(pa + a, pb + n - a).pdf
        pk_ = np.array([pk(x) for x in grid])
        ent = entropy(pk_, uniform.pdf)
    if isnan(ent) or isinf(ent):
        print('yield_kl_dist: ', a, n, pa + a, pb + n - a, ent)
    return ent


def get_wdist(f1, f2, grid):
    pk_ = np.array([f1(x) for x in grid])
    qk_ = np.array([f2(x) for x in grid])
    pk_ = pk_ / np.sum(pk_)
    qk_ = qk_ / np.sum(qk_)
    dist = np.zeros((pk_.size, qk_.size))
    for i in range(dist.shape[0]):
        for j in range(dist.shape[1]):
            dist[i, j] = abs(i - j) / pk_.size
    mx = pk_ / np.sum(pk_)
    my = qk_ / np.sum(qk_)
    plan = Variable((pk_.size, qk_.size))
    mx_trans = mx.reshape(-1, 1) * dist
    mu_trans_param = Parameter(mx_trans.shape, value=mx_trans)
    obj = Minimize(cvx_sum(cvx_mul(plan, mu_trans_param)))
    plan_i = cvx_sum(plan, axis=1)
    my_constraint = mx * plan
    constraints = [my_constraint == my,
                   plan >= 0, plan <= 1,
                   plan_i == np.ones(pk_.size)]
    problem = Problem(obj, constraints)
    solver = None
    solver_options = {}
    wd = problem.solve(solver=solver, **solver_options)
    return wd


def set_closest_max(x):
    log10 = np.log10(x)
    if log10 < 0:
        round_ind = int(log10) - 1
    else:
        round_ind = int(log10)
    xm = x / 10**round_ind
    x_up = round(xm + 0.5, 0)
    delta = x_up / xm - 1
    print('In set_closest_max() round_ind {}, delta {}, x_up {}, xm {}'.format(round_ind, delta, x_up, xm))
    if delta < 0.2:
        x_ans = x_up * 10 ** round_ind
    else:
        x_ans = (int(xm) + 1) * 10 ** round_ind
    return x_ans


def plot_thr_dt(df, fname=None):
    fig = plt.figure(figsize=(8, 8))
    rect = [0.15, 0.15, 0.75, 0.75]
    ax = fig.add_axes(rect)

    l1 = ax.plot(df.thr, df.dist_pos, color='b', alpha=0.8)
    l2 = ax.plot(df.thr, df.dist_neg, color='g', alpha=0.8)
    ax.set_ylabel('distance')
    ax.set_xlabel('threshold')
    ax2 = ax.twinx()
    l3 = ax2.plot(df.thr, df.n_pos, color='b', alpha=0.8, linestyle=':')
    l4 = ax2.plot(df.thr, df.n_neg, color='g', alpha=0.8, linestyle=':')
    ax2.set_ylabel('number')

    lns = l1 + l2 + l3 + l4
    ax.legend(lns, ['positive to ambivalent', 'negative to ambivalent',
                    'n positive', 'n negative'], loc='upper center')

    ax_max = max([df.dist_pos.max(), df.dist_neg.max()])
    # print(ax_max, set_closest_max(ax_max))
    ax_max2 = max([df.n_pos.max(), df.n_neg.max()])
    ax.set_ylim([0, set_closest_max(ax_max)])
    ax2.set_ylim([0, set_closest_max(ax_max2)])
    if fname:
        x = plt.savefig(fname)
    plt.close()


def get_thr_study(df, delta=2e-3, min_thr=5e-2, max_thr=4e-1, thrs=None,
                  thr_column=cexp, vote_column=ps, positive_flag=True, verbose=False):
    acc_data = []
    if not thrs:
        thrs = np.arange(min_thr, max_thr, delta)
    for thr in thrs:
        if verbose:
            print('thr = {0:.3f}'.format(thr))
        if positive_flag:
            mask = (df[thr_column] > 1.0 - thr)
        else:
            mask = (df[thr_column] < thr)

        par_ps = df.loc[mask, vote_column].sum(), sum(mask)
        par_not_ps = df.loc[~mask, vote_column].sum(), sum(~mask)
        acc_data.append([thr, *par_ps, *par_not_ps])
    info_df = pd.DataFrame(acc_data, columns=['thr', 'k', 'n', 'k_ambi', 'n_ambi'])
    return info_df


def get_dists(beta_data_df, foo, gridn=100, alpha0=0.5, beta0=0.5, verbose=False):
    dists = []
    delta = 1. / gridn
    grid = np.arange(0.0, 1., delta)

    prev_item = -1, -1, -1, -1, -1
    for j, item in beta_data_df.iterrows():
        thr, k, n, k2, n2 = item[['thr', 'k', 'n', 'k_ambi', 'n_ambi']]
        if k == prev_item[1] and n == prev_item[2]:
            dist = dists[-1]
        else:
            pdfs = [beta(alpha0 + k, beta0 + n - k).pdf, beta(alpha0 + k2, beta0 + n2 - k2).pdf]
            dist = foo(*pdfs, grid)
            if verbose:
                print('{0}/{1}:  d (pos, ambi) = {2:.3f}'.format(j, beta_data_df.shape[0], dist))
        dists.append(dist)
        prev_item = thr, k, n, k2, n2
    s = pd.Series(dists, index=beta_data_df.index)
    beta_data_df['dist'] = s
    return beta_data_df


origin = 'gw'
version = 11
cooked_version = 12

df0 = pd.read_csv('~/data/kl/final/thr_{0}_{1}_{2}.csv.gz'.format(origin, version, cooked_version),
                  compression='gzip', index_col=0)
origin = 'lit'
version = 8

df0lit = pd.read_csv('~/data/kl/final/thr_{0}_{1}_{2}.csv.gz'.format(origin, version, cooked_version),
                     compression='gzip', index_col=0)


df = df0.copy()
df_dict = {'gw': df0, 'lit': df0lit}

for key, df_ in df_dict.items():
    print(key, df_.shape)

for key, df_ in df_dict.items():
    if key == 'lit':
        mask_lit = (df_[up] == 7157) & (df_[dn] == 1026)
        print('filtering out 7157-1026 from lit: {0} rows out '.format(sum(mask_lit)))
        df_dict[key] = df_.loc[~mask_lit].copy()

for key, df_ in df_dict.items():
    print(key, df_.shape)


wdist_grid = 1000
thr_min = 2e-2
thr_max = 4e-1
thr_max = 2.4e-2
thr_delta = 2e-3

for k, df in df_dict.items():
    print(k, df.shape)
    df_info = get_thr_study(df, thr_delta, thr_min, thr_max)
    # print(df_info)
    df_info = get_dists(df_info, get_wdist, wdist_grid, verbose=True)
    # print(df_info)
    # df_info = df_info.rename(columns={'dist': 'ps', 'n': 'n_ps', 'k': 'k_ps'})
    # print(df_info)

    df_info2 = get_thr_study(df, thr_delta, thr_min, thr_max, positive_flag=False)
    # print(df_info2)
    df_info2 = get_dists(df_info2, get_wdist, wdist_grid, verbose=True)
    # print(df_info2)
    # df_info2 = df_info2.rename(columns={'dist': 'neg', 'n': 'n_neg'})
    # print(df_info2)
    dfr = pd.merge(df_info[['thr', 'dist', 'k', 'n']], df_info2[['thr', 'dist', 'k', 'n']], on='thr',
                   suffixes=['_pos', '_neg'])
    # print(dfr)
    fname = expanduser('~/data/kl/figs/{0}_thr_bdist_t.pdf'.format(k))
    plot_thr_dt(dfr, fname)
    dfr.to_csv(expanduser('~/data/kl/threshold/{0}_thr_t.csv'.format(k)))
