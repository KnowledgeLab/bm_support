from datahelpers.community_tools import calculate_comms
from itertools import product


methods = ['multilevel', 'infomap']
directeds = [True, False]
weighteds = [True, False]
percentile_values = [None, 95]

keys = ['method', 'directed']
keys2 = ['weighted', 'percentile_value']
largs = [{k: v for k, v in zip(keys, p)} for p in product(*(methods, directeds))]

zargs = [{k: v for k, v in zip(keys2, p)} for p in zip(*(weighteds, percentile_values))]

inv_args = {'full_fname': '~/data/lincs/graph/adj_mat_all_pert_types.h5',
            'fpath_out': '~/data/kl/comms/',
            'file_format': 'matrix'}
targs = [{**z, **l, **inv_args} for l, z in product(largs, zargs)]

targs2 = list(filter(lambda x: not (x['directed'] is False and x['method'] == 'multilevel'), targs))
targs2 = list(filter(lambda x: x['directed'] is False and x['percentile_value'], targs2))
print(targs2)

for args in targs2:
    dt = calculate_comms(**args)
    print(args)
    print('calc took {0:.2f} sec'.format(dt))
