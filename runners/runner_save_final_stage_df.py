from datahelpers.constants import ye, ai, ps, up, dn, ar, ni, dist, rdist, pm, ct, affs, aus
from bm_support.supervised import generate_samples
from bm_support.add_features import prepare_final_df
from os.path import expanduser

an_version = 12

origin = 'gw'
version = 11
hash_int = 471980
# origin = 'lit'
# version = 8
# hash_int = 502784

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

cols_active = [ai, ar, ct] + cur_metric_columns_exp + \
              ['pre_' + aus, 'nhi_' + aus, 'pre_' + affs, 'nhi_' + affs, 'year_off', 'year_off2']

eps = 0.2
upper_exp, lower_exp = 1 - eps, eps
thrs = [-1e-8, lower_exp, upper_exp, 1.0001e0]
aff_dict_fname = expanduser('~/data/wos/affs_disambi/pm2id_dict.pgz')

df2_ = prepare_final_df(df, normalize=True, columns_normalize=cols_norm,
                        columns_normalize_by_interaction=cols_norm_by_int,
                        quantize_intervals=thrs, aff_dict_fname=aff_dict_fname,
                        masks=masks, cutoff=None,
                        add_cite_fits=True, define_visible_prior=True,
                        verbose=True)


print('df2_ size {0}, df2_ unique [up, dn, pm] {1}'.format(df2_.shape[0],
                                                           df2_.drop_duplicates([up, dn, pm]).shape[0]))
df2_ = df2_.drop_duplicates([up, dn, pm])

# df2_.to_hdf(expanduser('~/data/kl/final/{0}_{1}_{2}.h5'.format(origin, version, an_version)), key='df', complevel=9)
