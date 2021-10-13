import pandas as pd
from pandas import HDFStore
from os.path import expanduser
from datahelpers.community_tools import prepare_graphdf, prepare_graph_from_df

from itertools import product
from pprint import pprint


def extract_order(y, upx, dnx, nets):
    if y in map_prev_year.keys():
        prev_year = map_prev_year[y]
        g, invmap, conv_map, _, _ = nets[prev_year]
        if upx in conv_map.keys():
            upn = conv_map[upx]
            up_deg_in, up_deg_out = g.vs[upn].degree("int"), g.vs[upn].degree("out")
        else:
            up_deg_in, up_deg_out = 0, 0
        if dnx in conv_map.keys():
            dnn = conv_map[dnx]
            dn_deg_in, dn_deg_out = g.vs[dnn].degree("int"), g.vs[dnn].degree("out")
        else:
            dn_deg_in, dn_deg_out = 0, 0
    else:
        up_deg_in, up_deg_out = 0, 0
        dn_deg_in, dn_deg_out = 0, 0
    return up_deg_in, up_deg_out, dn_deg_in, dn_deg_out


dfye = pd.read_csv(
    expanduser("~/data/kl/comms/interaction_network/updn_years2.csv.gz"), index_col=0
)


df_type = "lit"
versions = [8, 11]
df_types = ["lit", "gw"]
run_over_keys_flag = True
verbosity = False

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


ttargs = targs3[2]
pprint(ttargs)

full_fname = ttargs["full_fname"]

set_keys = set()
for f in full_fname:
    store = HDFStore(expanduser(f), mode="r")
    set_keys |= set(store.keys())
    store.close()
# keys = sorted(list(set_keys))
keys = sorted(list(set_keys))[:5]
tot = 0

nets = {}

# prepare graphs by year
for k in keys:
    print(k)
    df = prepare_graphdf(full_fname, ttargs["file_format"], k)
    g, ws, inv_conversion_map = prepare_graph_from_df(
        df, ttargs["file_format"], ttargs["directed"], ttargs["percentile_value"]
    )
    geneid2int = {v: k for k, v in inv_conversion_map.items()}

    kk = int(k[2:])
    # transform "/y1999" to 1999
    nets[kk] = g, inv_conversion_map, geneid2int, df, ws


sorted_years = sorted(nets.keys())

map_prev_year = dict(zip(sorted_years[1:], sorted_years[:]))

dgs = []
for ii, item in dfye.iterrows():
    upx, dnx, y = item[["up", "dn", "year"]].values
    degs = extract_order(y, upx, dnx, nets)
    dgs.append((upx, dnx, y, *degs))

dfd = pd.DataFrame(
    dgs,
    columns=[
        "up",
        "dn",
        "year",
        "degree_source_in",
        "degree_source_out",
        "degree_target_in",
        "degree_target_out",
    ],
)

dfd["degree_source"] = dfd["degree_source_in"] + dfd["degree_source_out"]
dfd["degree_target"] = dfd["degree_target_in"] + dfd["degree_target_out"]

# dfd.to_csv(expanduser('~/data/kl/comms/interaction_network/updn_degrees_directed.csv.gz'),
#            compression='gzip')
