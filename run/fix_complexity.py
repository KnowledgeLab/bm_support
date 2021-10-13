import argparse
from bm_support.add_features import define_laststage_metrics, prepare_datasets
from os.path import expanduser, join
from numpy.random import RandomState
import numpy as np
from bm_support.supervised_aux import run_model_iterate_over_datasets
from bm_support.math import interpolate_nonuniform_linear, integral_linear
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def savefigs(
    report_, pred_mode, master_col, xlabel, fpath_figs=expanduser("~/data/kl/figs/tmp")
):

    print(f"mode: {pred_mode}")

    acc = []
    for depth, items in report_.items():
        for items2 in items:
            for item in items2:
                it, origin, j, dfs, clf, mdict, coeffs, cfeats = item
                # print(mdict['auc'], mdict['train_report']['auc'])
                fpr, tpr, _ = mdict["roc_curve"]
                auc = integral_linear(fpr, tpr)
                acc.append((depth, origin, "test", auc))
                fpr, tpr, _ = mdict["train_report"]["roc_curve"]
                auc2 = integral_linear(fpr, tpr)
                acc.append((depth, origin, "train", auc2))

    df_acc = pd.DataFrame(acc, columns=[master_col, "origin", "sample", "value"])

    for origin in ["gw", "lit"]:
        fig = plt.figure(figsize=(8, 8))
        rect = [0.15, 0.15, 0.75, 0.75]
        ax = fig.add_axes(rect)
        df_acc2 = df_acc.loc[df_acc["origin"] == origin]
        sns.lineplot(data=df_acc2, x=master_col, y="value", hue="sample", ax=ax)
        plt.xlabel(xlabel)
        plt.ylabel("AUC ROC value")
        fig.savefig(join(fpath_figs, f"{pred_mode}_{origin}_auc_{master_col}.pdf"))
        fig.savefig(
            join(fpath_figs, f"{pred_mode}_{origin}_auc_{master_col}.png"),
            bbox_inches="tight",
            dpi=300,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m", "--mode", default="posneg", help="type of data to work with [gw, lit]"
    )

    parser.add_argument(
        "-n", "--niter", type=int, default=1, help="number of iterations"
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=13,
        help="seed, used to control random functions",
    )

    parser.add_argument(
        "-o",
        "--oversample",
        type=bool,
        default=False,
        help="use if your dataset is unbalanced",
    )

    min_leaf_frac_baseline = 0.005

    args = parser.parse_args()
    predict_mode = args.mode
    seed = args.seed
    n_iter = args.niter

    oversample = args.oversample

    print(f"mode: {predict_mode}")
    mode = "rf"
    rns = RandomState(seed)

    df_dict, cfeatures, target = prepare_datasets(predict_mode)

    verbose = False
    if predict_mode == "neutral":
        oversample = True
    else:
        oversample = False

    # ***
    # depth
    clf_parameters = {"max_depth": 6, "n_estimators": 100}
    extra_parameters = {"min_samples_leaf_frac": min_leaf_frac_baseline}

    depths = list(range(1, 7))
    sreport = {k: [] for k in depths}
    for cur_depth in depths:
        print(f"*** depth: {cur_depth}")
        clf_parameters["max_depth"] = cur_depth

        container = run_model_iterate_over_datasets(
            df_dict,
            cfeatures,
            rns,
            target=target,
            mode=mode,
            n_splits=3,
            clf_parameters=clf_parameters,
            extra_parameters=extra_parameters,
            n_iterations=n_iter,
            oversample=oversample,
        )

        sreport[cur_depth].append(container)

    savefigs(sreport, predict_mode, "depth", "depth of decision tree")

    # ***
    # min leaf size

    clf_parameters = {"max_depth": 2, "n_estimators": 100}
    min_leaves = [0.0005, 0.001, 0.025, 0.05, 0.01, 0.025, 0.05]
    sreport = {k: [] for k in min_leaves}

    for min_leaf in min_leaves:
        print(f"*** leaf_frac: {min_leaf}")

        extra_parameters = {"min_samples_leaf_frac": min_leaf}

        container = run_model_iterate_over_datasets(
            df_dict,
            cfeatures,
            rns,
            target=target,
            mode=mode,
            n_splits=3,
            clf_parameters=clf_parameters,
            extra_parameters=extra_parameters,
            n_iterations=n_iter,
            oversample=oversample,
        )

        sreport[min_leaf].append(container)

    savefigs(
        sreport, predict_mode, "leaf_frac", "min leaf size as fraction dataset size"
    )

    # ***
    # n estimators

    clf_parameters = {"max_depth": 1, "n_estimators": 100}
    estimators = np.arange(10, 150, 10)
    extra_parameters = {"min_samples_leaf_frac": min_leaf_frac_baseline}
    sreport = {k: [] for k in estimators}

    for x in estimators:
        print(f"*** n_est: {x}")
        clf_parameters["n_estimators"] = x

        container = run_model_iterate_over_datasets(
            df_dict,
            cfeatures,
            rns,
            target=target,
            mode=mode,
            n_splits=3,
            clf_parameters=clf_parameters,
            extra_parameters=extra_parameters,
            n_iterations=n_iter,
            oversample=oversample,
        )
        sreport[x].append(container)

    savefigs(sreport, predict_mode, "n_estimators", "number of estimators")
