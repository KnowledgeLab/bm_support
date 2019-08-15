from scipy.special import beta as beta_func
from scipy.special import gamma as gamma_func
import operator as op
from functools import reduce
from bm_support.supervised import report_metrics_, clean_zeros
from datahelpers.constants import up, dn, ps, cexp
import numpy as np
import pandas as pd


def transform_logp(x, barrier=20):
    y = max([-barrier, min([barrier, x])])
    return y


def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom


def beta_binomial(k, n, alpha, beta):
    return ncr(n, k)*beta_func(alpha + k, beta + n - k)/beta_func(alpha, beta)


def beta_binomial_real(k, n, alpha, beta):
    return (gamma_func(n+1)/(gamma_func(k+1)*gamma_func(n-k+1))) * \
           beta_func(alpha + k, beta + n - k)/beta_func(alpha, beta)


def produce_beta_est(df, cname, pars_pos, pars_neg, weighted=False, n_weight=10):
    bb = beta_binomial_real
    if weighted:
        df['weight'] = df['pi_pos'].apply(lambda x: int((n_weight*np.abs((x - 0.5)) / 0.5)))
        df_est = df.groupby([up, dn]).apply(lambda x: ((x[cname]*x['weight']).sum()/x['weight'].sum(), x.shape[0]))
    else:
        df_est = df.groupby([up, dn]).apply(lambda x: (x[cname].sum(), x.shape[0]))

    df_est_odds_ratio = df_est.apply(lambda x: (bb(*(list(x) + list(pars_pos))), bb(*(list(x) + list(pars_neg))),
                                                x[0] / x[1], *pars_pos, *pars_neg))

    df_pred = df_est_odds_ratio.apply(lambda x: 1./(1+x[1]/x[0])
                                      if ((x[0] != 0) and (x[1] != 0))
                                      and not np.isnan(x[0]) and not np.isnan(x[1])
                                      else int(abs(x[2] - x[3]) > abs(x[2] - x[4])))
    return df_pred


def produce_claim_valid(df, case_features_red, clf, pos_label=1, p_min=1e-2):
    # TESTED. pos_label is not needed
    # return pi_pos, prob of positive ps
    # print(df[case_features_red].shape)
    arr_probs = clf.predict_proba(df[case_features_red])
    arr_probs2 = clean_zeros(arr_probs, p_min)
    arr_ps = df[ps].values
    tensor_claims = np.stack([1 - arr_ps, arr_ps]).T
    if pos_label == 1:
        tensor_probs = np.stack([arr_probs2[:, ::-1], arr_probs2]).T
    else:
        tensor_probs = np.stack([arr_probs2, arr_probs2[:, ::-1]]).T
    # print('**', tensor_probs.shape, tensor_claims.shape)
    tt = np.sum(tensor_probs * tensor_claims, axis=2)
    # print('**', tt.shape)
    # print(tt[:, :5])
    df['correct'] = arr_probs[:, pos_label]
    df['pi_pos'] = tt[pos_label]
    return df


def produce_thrs(df, column2thr='pi_pos', thrs=np.arange(0.01, 0.9, 0.05), mean_flag=False, verbose=False):
    columns = []
    f_pos = df[ps].mean()
    thr = np.percentile(df[column2thr], 100 * (1 - f_pos))
    cname = '{0}_q{1}'.format(column2thr, 'obs')
    columns.append(cname)
    df[cname] = (df[column2thr] > thr).astype(int)
    if verbose:
        print('frac negs: {0:.3f}'.format(1 - f_pos))
    if mean_flag:
        thrs += (1 - f_pos)
    for f_pos in thrs:
        cname = '{0}_q{1:.1f}'.format(column2thr, 100*f_pos)
        thr = np.percentile(df[column2thr], 100*f_pos)
        df[cname] = (df[column2thr] > thr).astype(int)
        columns.append(cname)

    return df, columns


def produce_beta_interaction_est(df_train, df_test, columns=[ps, 'pi_pos'], verbose=False):
    mask = (df_train.bint == 1)
    pos_par = df_train.loc[mask, ps].sum(), sum(mask)
    neg_par = df_train.loc[~mask, ps].sum(), sum(~mask)
    pars_pos = pos_par[0], pos_par[1] - pos_par[0]
    pars_neg = neg_par[0], neg_par[1] - neg_par[0]

    if verbose:
        print('pos:', pars_pos)
        print('neg:', pars_neg)
        print(columns)
    y_pred_probs = [produce_beta_est(df_test, c, pars_pos, pars_neg) for c in columns]

    return y_pred_probs


def produce_interaction_plain(df, cname='pi_pos', weighted=False, n_weight=10):
    if weighted:
        center = df[ps].mean()
        df['weight'] = df['pi_pos'].apply(lambda x: (n_weight*np.abs(x - center) / 0.5))
        y_pred_probs = df.groupby([up, dn]).apply(lambda x: (x[cname]*x['weight']).sum()/x['weight'].sum())
    else:
        y_pred_probs = df.groupby([up, dn]).apply(lambda x: x[cname].mean())
    return y_pred_probs


def estimate_pi(df, cname='pi_pos', mode='plain'):
    if mode == 'plain':
        y_pred_probs = df.groupby([up, dn]).apply(lambda x: x[cname].mean())
    elif mode == 'rank':
        df['pct'] = df[cname].rank(pct=True)
        df['pct_mid_abs'] = (df['pct'] - 0.5).abs()**0.5
        y_pred_probs = df.groupby([up, dn]).apply(lambda x:
                                                  (x[cname]*x['pct_mid_abs']).sum()/x['pct_mid_abs'].sum()
                                                  if x['pct_mid_abs'].sum() > 0 else x[cname].mean())
    else:
        y_pred_probs = None
    return y_pred_probs


def produce_mu_est2(df_train, df_test, verbose=False):
    mus = df_test.groupby([up, dn]).apply(lambda x: x[ps].mean())
    return mus


def produce_interaction_est_weighted(df_train, df_test, columns=['pi_pos'], verbose=False):
    mask = (df_train.bint == 1)
    pos_par = df_train.loc[mask, ps].sum(), sum(mask)
    neg_par = df_train.loc[~mask, ps].sum(), sum(~mask)
    pars_pos = pos_par[0], pos_par[1] - pos_par[0]
    pars_neg = neg_par[0], neg_par[1] - neg_par[0]

    if verbose:
        print('pos:', pars_pos)
        print('neg:', pars_neg)
        print(columns)
    y_pred_probs = [produce_beta_est(df_test, c, pars_pos, pars_neg, weighted=True) for c in columns]

    return y_pred_probs


def produce_mu_est(df):
    int_cexp_est = df.groupby([up, dn]).apply(lambda x: x[ps].mean()).sort_index()
    int_cexp_est2 = df.groupby([up, dn]).apply(lambda x: x['pi_pos'].mean()).sort_index()
    return int_cexp_est, int_cexp_est2

