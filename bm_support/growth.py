from datahelpers.constants import ye, up, dn
from bm_support.sampling import assign_chrono_flag, fill_seqs
from bm_support.beta_est import produce_claim_valid
from bm_support.beta_est import estimate_pi
from bm_support.supervised_aux import produce_topk_model_
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
from functools import partial


def neg_pl_likelihood(gamma , cnts_dict):
    nstat_full = {k: cnts_dict[k] if k in cnts_dict.keys() else 0
                  for k in range(min(cnts_dict.keys()), max(cnts_dict.keys()) + 1)}
    xcoord = np.array(sorted(nstat_full.keys()), dtype=float)
    counts = np.array([nstat_full[k] for k in xcoord])
    probs = xcoord**(-gamma)
    probs = probs/np.sum(probs)
    LL = np.sum(counts*np.log(probs))
    return -LL


def check_distribution_dstructs(pdf_dict_imperfect, pdf_list_perfect):
    imperfect_lengths = [[k]*len(pdf_dict_imperfect[k]) for k in pdf_dict_imperfect.keys()]
    imperfect_lengths = [x for sublist in imperfect_lengths for x in sublist]
    perfect_lengths = [df['size'].sum() for df in pdf_list_perfect]
    # dfr = pd.DataFrame([(k, len(pdf_dict_imperfect[k])) for k in sorted_keys],
    #                    columns=['length', 'population'])
    return imperfect_lengths, perfect_lengths


def check_dstructs(pdf_dict_imperfect, pdf_list_perfect, wcolumn='accounted'):
    acc = [x for sublist in pdf_dict_imperfect.values() for x in sublist]
    if acc:
        dfg = pd.concat(acc)
        underfilled_size = dfg.loc[dfg[wcolumn], 'size'].sum()
        underfilled_potential_size = dfg['size'].sum()
    else:
        underfilled_size = 0
        underfilled_potential_size = 0
    if pdf_list_perfect:
        dfp = pd.concat(pdf_list_perfect)
        filled_size = dfp.loc[dfp[wcolumn], 'size'].sum()
    else:
        filled_size = 0
    return underfilled_size, underfilled_potential_size, filled_size


class SeqLenGrower:
    def __init__(self, df0, wcolumn='accounted',
                 init_frac=0.5,
                 fill_max=100,
                 index_cols=[up, dn],
                 time_column=ye,
                 verbose=False):

        """
        general idea :
            an elementary object is a sequence, which is labeled by (up, dn) // interaction index
                ((up, dn), y_1, s_1), ((up, dn), y_2, s_2), ((up, dn), y_3, s_3)

            we fill up the sequences chronologically and assign flags indicating
                whether the batch (up, dn), y_j will be considered in the estimation

                up    dn    year    size    accounted
                1543  2944  2008    2       True
                1543  2944  2009    3       False

            pdf_dict_imperfect is a dict with keys k
            and holds sequences which are full at level k (sum of size column)


        :param df0: DataFrame with
        :param wcolumn:
        :param init_frac:
        :param verbose:
        """

        self.index_cols = index_cols
        self.time_column = time_column
        full_index_cols = index_cols + [time_column]
        self.full_index_cols = full_index_cols

        # list of perfect seqs

        self.pdf_list_complete = []

        # dict of imperfect seqs
        # keys are lengths
        self.pdf_dict_incomplete = {}

        self.wcolumn = wcolumn
        self.df = df0.copy()
        self.verbose = verbose
        df_masker = df0.groupby(full_index_cols).apply(lambda x: x.shape[0])
        df_masker = df_masker.reset_index().rename(columns={0: 'size'})
        df_masked = df_masker.groupby(index_cols).apply(lambda x: assign_chrono_flag(x, frac=init_frac))

        for ii, group in df_masked.groupby(index_cols):
            n_filled = sum(group.loc[group[wcolumn], 'size'])
            if group[wcolumn].all():
                self.pdf_list_complete.append(group[full_index_cols + ['size', wcolumn]])
            else:
                if n_filled in self.pdf_dict_incomplete.keys():
                    self.pdf_dict_incomplete[n_filled].append(group[full_index_cols + ['size', wcolumn]])
                else:
                    self.pdf_dict_incomplete[n_filled] = [group[full_index_cols + ['size', wcolumn]]]

        # adjust

        sa, _, sb = check_dstructs(self.pdf_dict_incomplete, self.pdf_list_complete)
        self.imperfect_size = sa
        self.filled_size = sb

        if self.verbose:
            print(f'sa : {self.imperfect_size} sb: {self.filled_size}')

    def populate_seqs_(self, n_items=100, direction=min):
        self.pdf_dict_incomplete, self.pdf_list_complete = fill_seqs(self.pdf_dict_incomplete,
                                                                     self.pdf_list_complete, n_items,
                                                                     direction=direction)
        if self.pdf_dict_incomplete:
            dfa = pd.concat([x for sublist in self.pdf_dict_incomplete.values() for x in sublist])
            dfa = dfa.loc[dfa[self.wcolumn]]
        else:
            dfa = pd.DataFrame()
        if self.pdf_list_complete:
            dfb = pd.concat(self.pdf_list_complete)
        else:
            dfb = pd.DataFrame()
        sa, _, sb = check_dstructs(self.pdf_dict_incomplete, self.pdf_list_complete)
        self.imperfect_size = sa
        self.filled_size = sb
        if self.verbose:
            print(f'sa : {self.imperfect_size} sb: {self.filled_size}'
                  f' tot: {self.imperfect_size + self.filled_size}')
        return dfa, dfb

    def __str__(self):
        sa, samax, sb = check_dstructs(self.pdf_dict_incomplete, self.pdf_list_complete)
        return f'sa : {sa} sb: {sb} tot: {sa + sb} max: {sb + samax}'

    def get_pop_stats(self):
        underfilled_size, underfilled_potential_size, filled_size = \
            check_dstructs(self.pdf_dict_incomplete, self.pdf_list_complete)
        return underfilled_size, filled_size, underfilled_potential_size + filled_size

    def pop_populated_df(self, n_items=100, direction=min):
        dfa, dfb = self.populate_seqs_(n_items, direction=direction)
        if self.verbose:
            print(f'dfa : {dfa.shape[0]} dfb: {dfb.shape[0]}')
        dfc = pd.concat([dfa, dfb])
        dfr = pd.merge(self.df, dfc[self.full_index_cols], on=self.full_index_cols, how='right')
        return dfr


def populate_df(df, len_thr, cfeatures, clf, itarget, pop_delta=50, init_frac=0.2, direction=min,
                verbose=False):
    reports = []
    dfw = df[df.n > len_thr]
    df_len = dfw.shape[0]
    slg = SeqLenGrower(dfw, verbose=False, init_frac=init_frac)
    dfw = slg.pop_populated_df(0)

    while dfw.shape[0] < (1 - init_frac)*df_len:
        df_int = produce_claim_valid(dfw, cfeatures, clf)
        yi_test_ = pd.DataFrame(estimate_pi(df_int), columns=['muhat'])

        yi_test = df_int.drop_duplicates([up, dn]).sort_values([up, dn])[itarget]
        yi_pred_sc = yi_test_['muhat'].sort_index()
        if yi_test.unique().shape[0] == 2:
            report = produce_topk_model_(yi_test, yi_pred_sc)
            stats = slg.get_pop_stats()
            frac = (stats[0] + stats[1])/stats[2]
            nstat = dfw.groupby([up, dn]).apply(lambda x: x.shape[0])
            cnts_dict = dict(nstat.value_counts())
            foo = partial(neg_pl_likelihood, cnts_dict=cnts_dict)
            answer = minimize_scalar(foo, [0.1, 4.0])
            if answer.success:
                beta = answer.x
                reports += [(frac, nstat.mean(), nstat.std(), beta, report)]
        dfw = slg.pop_populated_df(pop_delta, direction=direction)
    return reports


def populate_df_chrono(df, len_thr, cfeatures, clf, itarget):
    reports = []
    dfwa = df[df.n > len_thr]

    years = sorted(dfwa[ye].unique())

    for yc in years[1:]:
        dfw = dfwa.loc[dfwa[ye] < yc].copy()
        df_int = produce_claim_valid(dfw, cfeatures, clf)
        yi_test_ = pd.DataFrame(estimate_pi(df_int), columns=['muhat'])

        yi_test = df_int.drop_duplicates([up, dn]).sort_values([up, dn])[itarget]
        yi_pred_sc = yi_test_['muhat'].sort_index()

        if yi_test.unique().shape[0] == 2:
            report = produce_topk_model_(yi_test, yi_pred_sc)
            frac = dfw.shape[0]/dfwa.shape[0]
            nstat = dfw.groupby([up, dn]).apply(lambda x: x.shape[0])
            cnts_dict = dict(nstat.value_counts())
            foo = partial(neg_pl_likelihood, cnts_dict=cnts_dict)
            answer = minimize_scalar(foo, [0.1, 4.0])
            if answer.success:
                beta = answer.x
                reports += [(frac, nstat.mean(), nstat.std(), beta, report, yc, dfw[ye].max())]
    return reports

