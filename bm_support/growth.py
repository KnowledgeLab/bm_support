from datahelpers.constants import iden, ye, ai, ps, up, dn
from bm_support.sampling import assign_half_flag, check_dstructs, fill_seqs
import pandas as pd


class SeqLenGrower:
    def __init__(self, df0, wcolumn='accounted', verbose=False):
        self.pdf_dict_perfect = []
        self.pdf_dict_imperfect = {}
        self.wcolumn = wcolumn
        self.df = df0.copy()
        self.verbose = verbose
        df_masker = df0.groupby([up, dn, ye]).apply(lambda x: x.shape[0])
        df_masker = df_masker.reset_index().rename(columns={0: 'size'})
        df_masked = df_masker.groupby([up, dn]).apply(lambda x: assign_half_flag(x))

        for ii, group in df_masked.groupby([up, dn]):
            n_filled = sum(group.loc[group[wcolumn], 'size'])
            if group[wcolumn].all():
                self.pdf_dict_perfect.append(group)
            else:
                if n_filled in self.pdf_dict_imperfect.keys():
                    self.pdf_dict_imperfect[n_filled].append(group[[up, dn, ye, 'size', wcolumn]])
                else:
                    self.pdf_dict_imperfect[n_filled] = [group[[up, dn, ye, 'size', wcolumn]]]

        sa, sb = check_dstructs(self.pdf_dict_imperfect, self.pdf_dict_perfect)
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
        sa, sb = check_dstructs(self.pdf_dict_imperfect, self.pdf_dict_perfect)
        self.imperfect_size = sa
        self.filled_size = sb
        if self.verbose:
            print(f'sa : {self.imperfect_size} sb: {self.filled_size}'
                  f' tot: {self.imperfect_size + self.filled_size}')
        return dfa, dfb

    def __str__(self):
        sa, sb = check_dstructs(self.pdf_dict_imperfect, self.pdf_dict_perfect)
        return f'sa : {sa} sb: {sb} tot: {sa + sb}'

    def pop_populated_df(self, n_items=100):
        dfa, dfb = self.populate_seqs_(n_items)
        if self.verbose:
            print(f'dfa : {dfa.shape[0]} dfb: {dfb.shape[0]}')
        dfc = pd.concat([dfa, dfb])
        dfr = pd.merge(self.df, dfc[[up, dn, ye]], on=[up, dn, ye], how='right')
        return dfr




