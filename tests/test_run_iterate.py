from os.path import expanduser, join
import pandas as pd
import json
import pickle
import gzip
from bm_support.supervised_aux import run_model_iterate
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
    set(gw_excl) | set(excl_set)
)
cfeatures = sorted(list(cfeatures))
print(cfeatures)

# mode = 'lr'
mode = "rf"

if mode == "rf":
    complexity_dict = {"max_depth": 4, "n_estimators": 100, "min_samples_leaf": 20}
else:
    complexity_dict = {"penalty": "l1", "solver": "liblinear", "max_iter": 100}

df = pd.concat(dfs)
target = "bdist"

rns = RandomState(13)

reports = run_model_iterate(
    df,
    cfeatures,
    rns,
    target,
    mode=mode,
    n_splits=3,
    clf_parameters=complexity_dict,
    n_iterations=2,
)
i0, j0, dfs_, clf, md, coefficients, cfeatures = reports[-1]
print(md["auc"], md["train_report"]["auc"])

dft1 = dfs_[1].copy()

rns = RandomState(13)
reports = run_model_iterate(
    df,
    cfeatures,
    rns,
    target,
    mode=mode,
    n_splits=3,
    clf_parameters=complexity_dict,
    n_iterations=2,
)
i0, j0, dfs_, clf, md, coefficients, cfeatures = reports[-1]
print(md["auc"], md["train_report"]["auc"])

dft2 = dfs_[1].copy()

print(all(dft1 == dft2))
