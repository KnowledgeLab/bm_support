import argparse
from bm_support.posterior_tools import fit_model_f
import gzip
import pickle
import logging
from os.path import join


def is_int(x):
    try:
        int(x)
    except:
        return False
    return True


def generate_fnames(modeltype, batch_size, j):
    return {
        "figname_prefix": "model_{0}_{1}_{2}".format(modeltype, batch_size, j),
        "tracename_prefix": "trace_{0}_{1}_{2}".format(modeltype, batch_size, j),
        "reportname_prefix": "report_{0}_{1}_{2}".format(modeltype, batch_size, j),
    }


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y" "1"):
        return True
    if v.lower() in ("no", "false", "f", "n" "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--batchsize", default="200", help="Size of data batches")
    parser.add_argument(
        "-b", "--begin", default="0", help="begin (starting) index in the batch"
    )
    parser.add_argument(
        "-e",
        "--end",
        default="0",
        help="end-1 (last) index in the batch; defaults to end of list",
    )
    parser.add_argument(
        "-m",
        "--modeltype",
        default="identity_ai_hiai_pos",
        help="model type specified by",
    )
    parser.add_argument(
        "-n", "--numberdraws", default="1000", help="mcmc number of draws"
    )

    parser.add_argument(
        "-f",
        "--logfilename",
        default="../../../tmp/runner_timestep.log",
        help="log filename",
    )
    parser.add_argument("--dry", type=str2bool, default=False, help="select dry run")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, filename=args.logfilename)

    if is_int(args.batchsize):
        batchsize = int(args.batchsize)
    else:
        raise ValueError("batchsize not an integer")

    if is_int(args.begin):
        begin = int(args.begin)
    else:
        raise ValueError("begin not an integer")

    if is_int(args.end):
        end = int(args.end)
    else:
        raise ValueError("end not an integer")

    if is_int(args.numberdraws):
        numberdraws = int(args.numberdraws)
    else:
        raise ValueError("numberdraws not an integer")
    modeltype = args.modeltype

    logging.info("batchsize : {0}".format(batchsize))
    logging.info("begin : {0}".format(begin))
    logging.info("end : {0}".format(end))
    logging.info("numberdraws : {0}".format(numberdraws))
    logging.info("modeltype : {0}".format(modeltype))

    logging.info("opening data_batches_{0}_{1}.pgz".format(modeltype, batchsize))
    with gzip.open(
        "../../../data/data_batches_{0}_{1}.pgz".format(modeltype, batchsize)
    ) as fp:
        dataset = pickle.load(fp)

    logging.info("dataset contains {0} items".format(len(dataset)))

    if end == 0:
        dataset = dataset[begin:]
        rr = range(begin, len(dataset))
    elif end <= len(dataset):
        dataset = dataset[begin:end]
        rr = range(begin, end)
    else:
        raise ValueError("end index is out of bounds of dataset")

    logging.info("cut dataset contains {0} items".format(len(dataset)))

    n_tot = numberdraws
    n_watch = int(0.9 * n_tot)
    n_step = 10

    barebone_dict_pars = {
        "n_features": 2,
        "fig_path": "./../../../figs/",
        "trace_path": "./../../../traces/",
        "report_path": "./../../../reports/",
        "n_total": n_tot,
        "n_watch": n_watch,
        "n_step": n_step,
        "plot_fits": True,
        "dry_run": args.dry,
    }

    kwargs_list = [
        {
            **barebone_dict_pars,
            **generate_fnames(modeltype, batchsize, j),
            **{"data_dict": d},
        }
        for j, d in zip(rr, dataset)
    ]

    results_list = map(lambda kwargs: fit_model_f(**kwargs), kwargs_list)

    reports_list = list(map(lambda x: x[0], results_list))

    with gzip.open(
        join("./../../../reports/", "reports_{0}_{1}.pgz".format(modeltype, batchsize)),
        "wb",
    ) as fp:
        pickle.dump(reports_list, fp)
