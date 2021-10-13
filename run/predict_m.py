import argparse
from bm_support.add_features import define_laststage_metrics, prepare_datasets
from os.path import expanduser, join
from numpy.random import RandomState
import pandas as pd
from bm_support.supervised_aux import run_model_iterate_over_datasets
import pickle
import gzip


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--model-class",
        default="posneg",
        help="model class: posneg, neutral, claims",
    )

    parser.add_argument("-m", "--model-type", default="rf", help="rf or lr")

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

    parser.add_argument("-t", "--thr", type=int, default=0, help="threshold in length")

    parser.add_argument("-p", "--pfeatures", type=int, default=0, help="features spec")

    parser.add_argument("-v", "--verbose", type=bool, default=False, help="verbosity")

    parser.add_argument("--known-aff", action="store_true", help="known aff")

    parser.add_argument("--top-journals", action="store_true", help="top_journals")

    parser.add_argument(
        "--version", type=int, default=0, help="set version for current run"
    )

    min_leaf_frac_baseline = 0.005

    args = parser.parse_args()
    model_class = args.model_class
    seed = args.seed
    len_thr = args.thr
    n_iter = args.niter
    verbose = args.verbose
    model_type = args.model_type

    version = args.version

    oversample = args.oversample

    print(f"model_class: {model_class}")
    rns = RandomState(seed)

    df_dict, cfeatures, target = prepare_datasets(
        model_class, len_thr, known_aff=args.known_aff, top_journals=args.top_journals
    )

    if model_class == "neutral":
        oversample = True
    else:
        oversample = False

    report = []
    if model_type == "rf":
        if model_class == "claims":
            clf_parameters = {
                "max_depth": 4,
                "n_estimators": 100,
                "min_samples_leaf": 20,
            }
        else:
            clf_parameters = {"max_depth": 2, "n_estimators": 100}
        extra_parameters = {"min_samples_leaf_frac": min_leaf_frac_baseline}
    else:
        clf_parameters = {"penalty": "l1", "solver": "liblinear", "max_iter": 100}
        extra_parameters = {"max_features": 7}

    if args.pfeatures:
        dff = pd.read_csv("~/data/kl/importances/best_features.csv", index_col=0)
        m = dff.origin == "lit"
        top_features = dff.loc[m, "f"].unique()
        cfeatures = [f for f in cfeatures if f in top_features]

    container = run_model_iterate_over_datasets(
        df_dict,
        cfeatures,
        rns,
        target=target,
        mode=model_type,
        clf_parameters=clf_parameters,
        extra_parameters=extra_parameters,
        n_splits=3,
        n_iterations=n_iter,
        oversample=oversample,
        verbose=verbose,
    )

    fpath = expanduser("~/data/kl/reports/")

    with gzip.open(
        fpath
        + f"models_{model_class}_mt_{model_type}_thr_{len_thr}_n_{n_iter}_v{version}.pkl.gz",
        "wb",
    ) as f:
        pickle.dump(container, f)
