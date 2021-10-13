from os.path import expanduser, join
import pandas as pd
import json
import pickle
import gzip
from bm_support.supervised_aux import run_model_, run_model_iterate
from numpy.random import RandomState


fname = expanduser("~/data/kl/columns/feature_groups_v3.txt")
with open(fname, "r") as f:
    feat_selector = json.load(f)

cfeatures0 = list(set(feat_selector["claim"]) | set(feat_selector["batch"]))


version = 8
fpath = expanduser("~/data/kl/reports/")
with gzip.open(join(fpath, f"models_claims_v{version}.pkl.gz"), "rb") as fp:
    container8 = pickle.load(fp)

key = "gw"
container_key = (item for item in container8 if item[1] == key)
it, k, j, dfs, clf, md, coeffs = next(container_key)

gw_excl = [c for c in cfeatures0 if sum(dfs[0][c].isnull()) > 0]
gw_excl += [c for c in cfeatures0 if sum(dfs[1][c].isnull()) > 0]

excl_set = set(["bdist_ma_None", "bdist_ma_2"])
cfeatures = (set(feat_selector["claim"]) | set(feat_selector["batch"])) - (
    set(gw_excl) | excl_set
)
cfeatures = sorted(list(cfeatures))

print(len(cfeatures))
df = pd.concat(dfs)
target = "bdist"


mode = "lr"
# mode = 'rf'

if mode == "rf":
    complexity_dict = {"max_depth": 4, "n_estimators": 100, "min_samples_leaf": 20}
else:
    complexity_dict = {"penalty": "l1", "solver": "liblinear", "max_iter": 100}

extra_parameters = {"min_samples_leaf_frac": 0.05}


rns = RandomState(13)
rnew = run_model_(
    dfs, cfeatures, rns, target, mode=mode, clf_parameters=complexity_dict
)
dfs_, clf, metrics_dict, coefficients, cfeatures = rnew
print(metrics_dict["auc"], metrics_dict["train_report"]["auc"])

rns = RandomState(13)
rnew = run_model_(
    dfs, cfeatures, rns, target, mode=mode, clf_parameters=complexity_dict
)
dfs_, clf, metrics_dict, coefficients, cfeatures = rnew
print(metrics_dict["auc"], metrics_dict["train_report"]["auc"])
