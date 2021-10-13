import pandas as pd
from os.path import expanduser
from datahelpers.constants import iden, ye, ai, ps, up, dn
from bm_support.derive_feature import derive_refutes_community_features

fpath = "~/data/wos/comm_metrics/"
dfps = pd.read_csv("~/data/wos/pmids/updnyearpmidps_all.csv.gz")
dfr = derive_refutes_community_features(
    dfps,
    fpath,
    windows=[2, 3, None],
    metric_sources=("affiliations", "authors", "past"),
    work_var=ps,
    verbose=True,
)
dfr.to_csv(
    expanduser("~/data/kl/comm_metrics/{0}_comm_window_ave.csv.gz".format(ps)),
    compression="gzip",
)
