from datahelpers.community_tools import calculate_comms
from itertools import product
from pprint import pprint

methods = ["multilevel", "infomap"]
directeds = [True, False]
weighteds = [True, False]
percentile_values = [None, 95]

keys = ["method", "directed"]
keys2 = ["weighted", "percentile_value"]
largs = [{k: v for k, v in zip(keys, p)} for p in product(*(methods, directeds))]

zargs = [{k: v for k, v in zip(keys2, p)} for p in zip(*(weighteds, percentile_values))]

inv_args = {
    "full_fname": "~/data/lincs/graph/adj_mat_all_pert_types.h5",
    "fpath_out": "~/data/kl/comms/",
    "file_format": "matrix",
}

targs = [{**z, **l, **inv_args} for l, z in product(largs, zargs)]

pprint(targs)

targs2 = list(
    filter(lambda x: not (x["directed"] and x["method"] == "multilevel"), targs)
)
pprint(targs2)
targs2 = list(
    filter(
        lambda x: not (
            not x["directed"] and x["weighted"] and x["method"] == "infomap"
        ),
        targs2,
    )
)
# targs2 = list(filter(lambda x: not x['weighted'] and x['method'] == 'multilevel',
#                      targs2))
# targs2 = list(filter(lambda x: x['method'] == 'multilevel',
#                      targs2))

for t in targs2:
    t.update({"verbose": True, "origin": "lincs"})

pprint(targs2)
for args in targs2:
    dt = calculate_comms(**args)
    pprint(args)
    pprint("calc took {0:.2f} sec".format(dt))
