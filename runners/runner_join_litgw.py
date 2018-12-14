from datahelpers.constants import iden, ye, ai, ps, up, dn, ar, ni, cexp, qcexp, nw, wi, dist, pm, ct, affs, aus
from bm_support.supervised import generate_samples
from bm_support.add_features import prepare_final_df
import pandas as pd
from os.path import expanduser

def isin_tuple(df, list_tuples):
    #TODO: very slow - optimize
    flgs = []
    for ii, row in df.iterrows():
        flgs.append(any([all([x_ == y_ for x_, y_ in zip(row, tup)]) for tup in list_tuples]))
    return pd.Series(flgs, index=df.index)


fpath = expanduser('~/data/kl/figs/rf')

an_version = 12

origin = 'gw'
version = 11
hash_int = 471980

batchsize = 1
cutoff_len = 1
a = 0.0
b = 1.0

lincs_type = '4'

feauture_cols = [ai, ar]

windows = [1, 2, 3]
cur_metric_columns = ['cpoprc', 'cdenrc', 'ksstrc']

cur_metric_columns_exp = cur_metric_columns + [c+str(w) for w in windows for c in cur_metric_columns]
cur_metric_columns_exp_normed = []

data_columns = [ni, pm, ye] + feauture_cols + cur_metric_columns_exp + cur_metric_columns_exp_normed + [ps]
print(data_columns)

df = generate_samples(origin, version, a, b, batchsize, cutoff_len, lincs_type=lincs_type,
                      data_columns=data_columns, hash_int=hash_int, verbose=True)
print('df size {0}, df unique [up, dn, pm] {1}'.format(df.shape[0],
                                                       df.drop_duplicates([up, dn, pm]).shape[0]))
df = df.drop_duplicates([up, dn, pm])

masks = []
# mask affiliation rating
# ar < 0 means pmid was not in the db
ar_mask = [ar, 0., lambda df_, th: (df_ >= th)]
masks.append(ar_mask)

# mask article influence
# ai equal to the top of value_counts() means that it was imputed
ai_mask = [ai, 0., lambda s, th: (s != s.value_counts().index[0])]
masks.append(ai_mask)

cols_norm = []
cols_norm_by_int = []

cols_active = [ai, ar, ct] + cur_metric_columns_exp +               ['pre_' + aus, 'nhi_' + aus, 'pre_' + affs, 'nhi_' + affs, 'year_off', 'year_off2']

eps = 0.2
upper_exp, lower_exp = 1 - eps, eps
thrs = [-1e-8, lower_exp, upper_exp, 1.0001e0]
aff_dict_fname = expanduser('~/data/wos/affs_disambi/pm2id_dict.pgz')

df2_ = prepare_final_df(df, normalize=True, columns_normalize=cols_norm,
                        columns_normalize_by_interaction=cols_norm_by_int,
                        quantize_intervals=thrs, aff_dict_fname=aff_dict_fname,
                        masks=masks, cutoff=None,
                        add_cite_fits=True,
                        verbose=True)

print('df2_ size {0}, df2_ unique [up, dn, pm] {1}'.format(df2_.shape[0],
                                                           df2_.drop_duplicates([up, dn, pm]).shape[0]))
df2_ = df2_.drop_duplicates([up, dn, pm])

an_version = 12

origin = 'lit'
version = 8
hash_int = 502784

batchsize = 1
cutoff_len = 1
a = 0.0
b = 1.0

lincs_type = '4'

feauture_cols = [ai, ar]

windows = [1, 2, 3]
cur_metric_columns = ['cpoprc', 'cdenrc', 'ksstrc']

cur_metric_columns_exp = cur_metric_columns + [c+str(w) for w in windows for c in cur_metric_columns]
cur_metric_columns_exp_normed = []

data_columns = [ni, pm, ye] + feauture_cols + cur_metric_columns_exp + cur_metric_columns_exp_normed + [ps]
print(data_columns)

df = generate_samples(origin, version, a, b, batchsize, cutoff_len, lincs_type=lincs_type,
                      data_columns=data_columns, hash_int=hash_int, verbose=True)
print('df size {0}, df unique [up, dn, pm] {1}'.format(df.shape[0],
                                                       df.drop_duplicates([up, dn, pm]).shape[0]))
df = df.drop_duplicates([up, dn, pm])

masks = []
# mask affiliation rating
# ar < 0 means pmid was not in the db
ar_mask = [ar, 0., lambda df_, th: (df_ >= th)]
masks.append(ar_mask)

# mask article influence
# ai equal to the top of value_counts() means that it was imputed
ai_mask = [ai, 0., lambda s, th: (s != s.value_counts().index[0])]
masks.append(ai_mask)

cols_norm = []
cols_norm_by_int = []

cols_active = [ai, ar, ct] + cur_metric_columns_exp +               ['pre_' + aus, 'nhi_' + aus, 'pre_' + affs, 'nhi_' + affs, 'year_off', 'year_off2']

eps = 0.2
upper_exp, lower_exp = 1 - eps, eps
thrs = [-1e-8, lower_exp, upper_exp, 1.0001e0]
aff_dict_fname = expanduser('~/data/wos/affs_disambi/pm2id_dict.pgz')

df3_ = prepare_final_df(df, normalize=True, columns_normalize=cols_norm,
                        columns_normalize_by_interaction=cols_norm_by_int,
                        quantize_intervals=thrs, aff_dict_fname=aff_dict_fname,
                        masks=masks, cutoff=None,
                        add_cite_fits=True,
                        verbose=True)

print('df3_ size {0}, df3_ unique [up, dn, pm] {1}'.format(df3_.shape[0],
                                                           df3_.drop_duplicates([up, dn, pm]).shape[0]))
df3_ = df3_.drop_duplicates([up, dn, pm])

dfu = pd.merge(df2_[[up, dn, pm, ps]], df3_[[up, dn, pm, ps]], on=[up, dn, pm], how='inner')
mask = (dfu[ps+'_x'] == dfu[ps+'_y'])
dfu.shape, sum(mask), sum(~mask)


ltups = [tuple(x) for x in dfu.loc[~mask, [up, dn, pm]].values]

m2 = isin_tuple(df2_[[up, dn, pm]], ltups)
m3 = isin_tuple(df3_[[up, dn, pm]], ltups)

df2_cut = df2_.loc[~m2].copy()
df3_cut = df3_.loc[~m3].copy()


sum(m2), sum(m3)

dft = pd.concat([df2_cut, df3_cut])


dft2 = dft.drop_duplicates([up, dn, pm])


df2_.shape[0] + df3_.shape[0] - 2*sum(m2)
number_unique_rows = df2_.shape[0] + df3_.shape[0] - dfu.shape[0] - sum(m2)
print('number of unique rows : expected -> {0}; derived -> {1}'.format(number_unique_rows, dft2.shape[0]))
dft2.to_hdf(expanduser('~/data/kl/final/{0}_{1}_{2}.h5'.format('litgw', 1, an_version)), key='df', complevel=9)

