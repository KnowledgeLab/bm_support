from datahelpers.community_tools import calculate_comms
from itertools import product

df_type = 'lit'
versions = [8, 11]
df_types = ['lit', 'gw']

fnames = [{'full_fname': '~/data/kl/comms/edges_{0}{1}.h5'.format(t, v)}
          for t, v in zip(df_types, versions)]


methods = ['multilevel', 'infomap']
directeds = [True, False]
weighteds = [True, False]
percentile_values = [None, 95]
keys = ['method', 'directed']
keys2 = ['weighted', 'percentile_value']
largs = [{k: v for k, v in zip(keys, p)} for p in product(*(methods, directeds))]
zargs = [{k: v for k, v in zip(keys2, p)} for p in zip(*(weighteds, percentile_values))]

inv_args = {'fpath_out': '~/data/kl/comms/',
            'file_format': 'edges'}
targs = [{**z, **l, **inv_args} for l, z in product(largs, zargs)]
targs2 = list(filter(lambda x: not (x['directed'] is True and x['method'] == 'multilevel'), targs))
targs3 = [{**x, **y, **inv_args} for x, y in product(fnames, targs2)]

print(targs3)

for args in targs3:
    dt = calculate_comms(**args)
    print(args)
    print('calc took {0:.2f} sec'.format(dt))