from datahelpers.constants import iden, ye, ai, ps, up, dn
from bm_support.sampling import assign_chrono_flag, fill_seqs
import pandas as pd


def check_dstructs(pdf_dict_imperfect, pdf_dict_perfect, wcolumn='accounted'):
    acc = [x for sublist in pdf_dict_imperfect.values() for x in sublist]
    if acc:
        dfg = pd.concat(acc)
        sg = dfg.loc[dfg[wcolumn], 'size'].sum()
        sgmax = dfg['size'].sum()
    else:
        sg = 0
        sgmax = 0
    if pdf_dict_perfect:
        dfp = pd.concat(pdf_dict_perfect)
        sp = dfp.loc[dfp[wcolumn], 'size'].sum()
    else:
        sp = 0
    return sg, sgmax, sp


class SeqLenGrower:
    def __init__(self, df0, wcolumn='accounted',
                 init_frac=0.5, verbose=False):
        # list of perfect seqs

        self.pdf_dict_perfect = []

        # dict of imperfect seqs
        # keys are lengths
        self.pdf_dict_imperfect = {}

        self.wcolumn = wcolumn
        self.df = df0.copy()
        self.verbose = verbose
        df_masker = df0.groupby([up, dn, ye]).apply(lambda x: x.shape[0])
        df_masker = df_masker.reset_index().rename(columns={0: 'size'})
        df_masked = df_masker.groupby([up, dn]).apply(lambda x: assign_chrono_flag(x, frac=init_frac))

        for ii, group in df_masked.groupby([up, dn]):
            n_filled = sum(group.loc[group[wcolumn], 'size'])
            if group[wcolumn].all():
                self.pdf_dict_perfect.append(group[[up, dn, ye, 'size', wcolumn]])
            else:
                if n_filled in self.pdf_dict_imperfect.keys():
                    self.pdf_dict_imperfect[n_filled].append(group[[up, dn, ye, 'size', wcolumn]])
                else:
                    self.pdf_dict_imperfect[n_filled] = [group[[up, dn, ye, 'size', wcolumn]]]

        sa, _, sb = check_dstructs(self.pdf_dict_imperfect, self.pdf_dict_perfect)
        self.imperfect_size = sa
        self.filled_size = sb

        if self.verbose:
            print(f'sa : {self.imperfect_size} sb: {self.filled_size}')

    def populate_seqs_(self, n_items=100):
        self.pdf_dict_imperfect, self.pdf_dict_perfect = fill_seqs(self.pdf_dict_imperfect,
                                                                   self.pdf_dict_perfect, n_items)
        if self.pdf_dict_imperfect:
            dfa = pd.concat([x for sublist in self.pdf_dict_imperfect.values() for x in sublist])
            dfa = dfa.loc[dfa[self.wcolumn]]
        else:
            dfa = pd.DataFrame()
        if self.pdf_dict_perfect:
            dfb = pd.concat(self.pdf_dict_perfect)
        else:
            dfb = pd.DataFrame()
        sa, _, sb = check_dstructs(self.pdf_dict_imperfect, self.pdf_dict_perfect)
        self.imperfect_size = sa
        self.filled_size = sb
        if self.verbose:
            print(f'sa : {self.imperfect_size} sb: {self.filled_size}'
                  f' tot: {self.imperfect_size + self.filled_size}')
        return dfa, dfb

    def __str__(self):
        sa, samax, sb = check_dstructs(self.pdf_dict_imperfect, self.pdf_dict_perfect)
        return f'sa : {sa} sb: {sb} tot: {sa + sb} max: {sb + samax}'

    def get_pop_fracs(self):
        sa, samax, sb = check_dstructs(self.pdf_dict_imperfect, self.pdf_dict_perfect)
        return sa, sb, samax+sb

    def pop_populated_df(self, n_items=100):
        dfa, dfb = self.populate_seqs_(n_items)
        if self.verbose:
            print(f'dfa : {dfa.shape[0]} dfb: {dfb.shape[0]}')
        dfc = pd.concat([dfa, dfb])
        dfr = pd.merge(self.df, dfc[[up, dn, ye]], on=[up, dn, ye], how='right')
        return dfr




