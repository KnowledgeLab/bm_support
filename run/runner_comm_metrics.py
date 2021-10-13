from datahelpers.community_tools import meta_calculate_comms
from itertools import product
from pprint import pprint

df_type = "lit"
versions = [8, 11]
df_types = ["lit", "gw"]
# run_over_keys_flag = False
run_over_keys_flag = True
verbosity = False
# verbosity = True

multi_ys = [run_over_keys_flag] * len(df_types)
if run_over_keys_flag:
    fnames = [
        {
            "full_fname": "~/data/kl/comms/edges_all_{0}{1}.h5".format(t, v),
            "origin": t,
            "multy_years": my,
        }
        for t, v, my in zip(df_types, versions, multi_ys)
    ]
else:
    fnames = [
        {"full_fname": "~/data/kl/comms/edges_{0}{1}.h5".format(t, v), "origin": t}
        for t, v in zip(df_types, versions)
    ]

keys = ["origin", "full_fname"]
# create arguments for the merged lit/gw run

aggs = {k: [f[k] for f in fnames] for k in keys}
aggs["origin"] = "".join(aggs["origin"])
aggs["run_over_keys_flag"] = run_over_keys_flag

if run_over_keys_flag:
    fnames = [aggs]
else:
    fnames.append(aggs)

methods = ["multilevel", "infomap"]
directeds = [True, False]
weighteds = [True, False]
percentile_values = [None, 95]
keys = ["method", "directed"]
keys2 = ["weighted", "percentile_value"]
largs = [{k: v for k, v in zip(keys, p)} for p in product(*(methods, directeds))]
zargs = [{k: v for k, v in zip(keys2, p)} for p in zip(*(weighteds, percentile_values))]

inv_args = {"fpath_out": "~/data/kl/comms/", "file_format": "edges"}
targs = [{**z, **l, **inv_args} for l, z in product(largs, zargs)]
targs2 = list(
    filter(lambda x: not (x["directed"] is True and x["method"] == "multilevel"), targs)
)
targs3 = [{**x, **y, **inv_args} for x, y in product(fnames, targs2)]

# pprint(targs3)
# targs3 = list(filter(lambda x: not x['weighted'] and x['method'] == 'multilevel',
#                      targs3))
# targs3 = list(filter(lambda x: x['method'] == 'multilevel',
#                      targs3))
pprint(targs3)

for args in targs3[:]:
    dt = meta_calculate_comms(**{**args, **{"verbose": verbosity}})
    print(args)
    print("calc took {0:.2f} sec".format(dt))
