import gzip
import pickle
from os.path import expanduser
import pandas as pd
from datahelpers.constants import ye, up, dn

df_type = 'lit'
versions = [8, 11]
df_types = ['lit', 'gw']
version = 11

dfs = []
for ty, v in zip(df_types, versions):
    with gzip.open(expanduser('~/data/kl/claims/df_{0}_{1}.pgz'.format(ty, v)), 'rb') as fp:
        df = pickle.load(fp)
        dfs.append(df)


for ty, v, df in zip(df_types, versions, dfs):
    years = sorted(df[ye].unique())
    print(len(years))
    h5_fname = expanduser('~/data/kl/comms/edges_all_{0}{1}.h5'.format(ty, v))
    store = pd.HDFStore(h5_fname)
    for y in years[:]:
        mask = (df[ye] <= y)
        print(df[mask].shape)
        dfe = df.loc[mask].groupby([up, dn]).apply(lambda x: x.shape[0])
        store.put('y{0}'.format(y), dfe.reset_index(), format='t')
    store.close()
