from os.path import expanduser, join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pickle
import gzip
import seaborn as sns
from bm_support.supervised_aux import plot_auc
from bm_support.math import interpolate_nonuniform_linear, integral_linear

gamma = 1.96


def get_container(model_type="rf", model_class="neutral", len_thr=0, n_iter=20):
    fpath = expanduser("~/data/kl/reports/")
    with gzip.open(
        join(
            fpath,
            f"models_{model_class}_mt_{model_type}_thr_{len_thr}_n_{n_iter}_v{version}.pkl.gz",
        ),
        "rb",
    ) as fp:
        container_ = pickle.load(fp)
    return container_


mclasses = ["posneg", "neutral", "claims"]
model_type = "rf"
# mt = 'lr'
len_thr = 0
n_iter = 20
version = 7
# version = 133

pal = sns.color_palette()
pp = pal.as_hex()
cblue = pp[0]
corange = pp[1]
cgreen = pp[2]
cred = pp[3]

pal = sns.color_palette()
blues = sns.color_palette("Blues").as_hex()
reds = sns.color_palette("Reds").as_hex()
cpals = {"gw": blues, "lit": reds}
cpals_imp = {
    "gw": dict(zip([0, 1], blues[-3:][::-1])),
    "lit": dict(zip([0, 1], reds[-3:][::-1])),
}


def plot_aucs(container, model_class):
    fpath_figs = expanduser("~/data/kl/figs/")

    origins = ["gw", "lit"]
    acc = []

    for key in origins:
        ax = None
        n_disc = 100
        base_fpr = np.linspace(0, 1, n_disc + 1)
        tprs = []
        aucs = []
        plot_baseline = True

        container_key = (item for item in container if item[1] == key)
        for it, k, j, dfs, clf, md, coeffs, _ in container_key:
            ax = plot_auc(
                md,
                ax=ax,
                show_legend=False,
                plot_baseline=plot_baseline,
                alpha=0.1,
                color=cpals[key][-2],
                cbaseline="k",
            )
            plot_baseline = False

            fpr, tpr, _ = md["roc_curve"]
            tprn = interpolate_nonuniform_linear(base_fpr, fpr, tpr)
            tprs.append(tprn)
            aucs.append(integral_linear(fpr, tpr))

        tprs = np.array(tprs)
        mean_tprs = tprs.mean(axis=0)
        std = tprs.std(axis=0)
        std = gamma * std
        std = gamma * std / np.sqrt(len(aucs))

        tprs_upper = np.minimum(mean_tprs + std, 1)
        tprs_lower = mean_tprs - std
        ax.set_title(f"ROC: {key}, {model_class} model, {model_type}")

        ax.fill_between(
            base_fpr,
            tprs_lower,
            tprs_upper,
            color=cpals[key][-2],
            alpha=0.2,
            linewidth=1.5,
        )
        ll = ax.plot(base_fpr, mean_tprs, c=cpals[key][-1])
        ax.legend(
            ll,
            ["AUC score: " f"{np.mean(aucs):.3f}" r"$\pm$" f"{np.std(aucs):.3f}"],
            loc="lower right",
        )
        axins = inset_axes(
            ax,
            width="100%",
            height="100%",
            bbox_to_anchor=(0.65, 0.15, 0.3, 0.3),
            bbox_transform=ax.transAxes,
        )
        sns_ret = sns.distplot(
            aucs, hist=False, ax=axins, kde_kws={"color": cpals[key][-1], "shade": True}
        )
        ####
        # kdex, kdey = sns_ret.get_lines()[0].get_data()
        # mean_ind = sum(kdex < np.mean(aucs))
        # alpha = (kdex[mean_ind] - np.mean(aucs)) / (kdex[mean_ind] - kdex[mean_ind - 1])
        # ytop = (1 - alpha) * kdey[mean_ind - 1] + alpha * kdey[mean_ind]
        # hh, _ = np.histogram(aucs)
        # axins.plot(
        #     [np.mean(aucs), np.mean(aucs)],
        #     [0, 0.98 * ytop],
        #     c="k",
        #     linestyle="--",
        #     linewidth=2,
        # )
        # axins.set_xlim([0.5, 1.0])
        ###
        fname = f"{model_class}_mt_{model_type}_{key}_v{version}_auc"
        plt.savefig(fpath_figs + fname + ".pdf")
        plt.savefig(fpath_figs + fname + ".png", bbox_inches="tight", dpi=300)

        acc += [
            (
                key,
                model_class,
                np.mean(aucs),
                np.std(aucs),
                np.percentile(aucs, 5),
                np.percentile(aucs, 95),
            )
        ]
    return acc


agg = []

for mc in mclasses[:]:
    container = get_container(model_type, mc, len_thr, n_iter)
    r = plot_aucs(container, mc)
    agg += r

df = pd.DataFrame(agg, columns=["dataset", "mc", "mean", "std", "5%", "95%"])
df["version"] = version
df.to_csv(f"~/data/kl/reports/aucs/{version}.csv")
with pd.option_context("precision", 3):
    print(df)
