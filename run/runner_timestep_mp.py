import argparse
import multiprocessing as mp
from bm_support.posterior_tools import fit_model_f, fit_model_e
import gzip
import pickle
import logging
from os.path import join
import sys
from os.path import expanduser


function_dict = {"model_f": fit_model_f, "model_e": fit_model_e}


def is_int(x):
    try:
        int(x)
    except:
        return False
    return True


def generate_fnames(prefix_str, j):
    suffix = "{0}_it_{1}".format(prefix_str, j)
    return {
        "figname_prefix": "fig_{0}".format(suffix),
        "tracename_prefix": "trace_{0}".format(suffix),
        "reportname_prefix": "report_{0}".format(suffix),
    }


def decorate_wlogger(f, qu, log_fname):
    def foo(it, list_kwargs):
        logger_ = mp.get_logger()
        if log_fname != "stdout":
            lfn = "{0}_{1}{2}".format(log_fname[:-4], it, log_fname[-4:])
            file_handler = logging.FileHandler(lfn, mode="w")
            logger_.addHandler(file_handler)
            logger_.info("logfilename: {0}".format(lfn))
        logger_.info("start f: j {0}".format(it))
        r = list(map(lambda kwargs: f(**kwargs), list_kwargs))
        qu.put(r)

    return foo


def setup_logger(name, log_file, level=logging.INFO):

    handler = logging.FileHandler(log_file)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y" "1"):
        return True
    if v.lower() in ("no", "false", "f", "n" "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s", "--nsamples", default="8", type=int, help="Size of data batches"
    )
    parser.add_argument(
        "-b",
        "--begin",
        default="0",
        type=int,
        help="begin (starting) index in the batch",
    )
    parser.add_argument(
        "-e",
        "--end",
        default="0",
        type=int,
        help="end-1 (last) index in the batch; defaults to end of list",
    )

    parser.add_argument(
        "-d",
        "--data-columns",
        nargs="*",
        default=["year", "identity", "ai", "aihi", "pos"],
    )

    parser.add_argument(
        "-n", "--numberdraws", default="1000", type=int, help="mcmc number of draws"
    )

    parser.add_argument(
        "-p", "--nparallel", default="1", type=int, help="number of parallel threads"
    )

    parser.add_argument(
        "-f", "--logfilename", default="runner_timestep_mp.log", help="log filename"
    )

    parser.add_argument("--dry", type=str2bool, default=False, help="select dry run")

    parser.add_argument(
        "--datapath", default=expanduser("~/data/kl/batches"), help="data filepath"
    )

    parser.add_argument(
        "--reportspath",
        default=expanduser("~/data/kl/reports"),
        help="reports filepath",
    )

    parser.add_argument("--logspath", default="../../../logs", help="logs filepath")

    parser.add_argument("--figspath", default="../../../figs", help="figs filepath")

    parser.add_argument(
        "--tracespath", default="../../../traces", help="traces filepath"
    )

    parser.add_argument(
        "--func",
        default="model_f",
        help="which function to use for inference from : "
        "{0}".format(list(function_dict.keys())),
    )

    parser.add_argument(
        "--version", default=9, type=int, help="version of the data transformation"
    )

    parser.add_argument("--case", default="a", help="test case specifier")

    parser.add_argument(
        "--origin",
        default="gw",
        help="origin of the dataset, gw for geneways, lit for literome",
    )

    parser.add_argument(
        "--partition-interval", nargs="+", default=[0.5, 0.5], type=float
    )
    parser.add_argument("--index-interval", default=-1, type=int)

    parser.add_argument(
        "--minsize-sequence", default=20, type=int, help="version of data source"
    )

    parser.add_argument(
        "--partition-sequence",
        nargs="+",
        default=[0.1, 0.9],
        type=float,
        help="define interval of observed freqs for sequence consideration",
    )

    args = parser.parse_args()

    data_cols = "_".join(args.data_columns)
    # logger = setup_logger('main', args.logfilename, level=logging.INFO)

    if args.logfilename == "stdout":
        logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    else:
        logging.basicConfig(
            level=logging.INFO,
            filename=join(args.logspath, args.logfilename),
            filemode="w",
        )

    logging.info(args._get_kwargs())

    if not args.func in function_dict.keys():
        raise ValueError("specified inference function (--func) is not valid")

    logging.info("logger {0} at {1}".format(args.logfilename, args.logspath))
    logging.info("{0} threads will be started".format(args.nparallel))

    logging.info("nsamples : {0}".format(args.nsamples))
    logging.info("begin : {0}".format(args.begin))
    logging.info("end : {0}".format(args.end))
    logging.info("numberdraws : {0}".format(args.numberdraws))
    logging.info("data columns : {0}".format(data_cols))

    low_f, hi_f = args.partition_sequence
    n = args.minsize_sequence

    logging.info(
        "opening data_batches_{0}_v_{1}_c_{2}_m_{3}_n_{4}_a_{5}_b_{6}.pgz".format(
            args.origin, args.version, data_cols, args.nsamples, n, low_f, hi_f
        )
    )
    with gzip.open(
        join(
            args.datapath,
            "data_batches_{0}_v_{1}_c_{2}_m_{3}_n_{4}_a_{5}_b_{6}.pgz".format(
                args.origin, args.version, data_cols, args.nsamples, n, low_f, hi_f
            ),
        )
    ) as fp:
        dataset = pickle.load(fp)

    logging.info("dataset contains {0} items".format(len(dataset)))

    if args.end == 0:
        dataset = dataset[args.begin :]
        rr = range(args.begin, len(dataset))
    elif args.end <= len(dataset):
        dataset = dataset[args.begin : args.end]
        rr = range(args.begin, args.end)
    else:
        raise ValueError("end index is out of bounds of dataset")

    logging.info("cut dataset contains {0} items".format(len(dataset)))
    logging.info("dry run : {0}".format(args.dry))

    n_tot = args.numberdraws
    n_watch = int(0.9 * n_tot)
    n_step = 10
    n_features = len(args.data_columns) - 3

    barebone_dict_pars = {
        "n_features": n_features,
        "fig_path": args.figspath,
        "trace_path": args.tracespath,
        "report_path": args.reportspath,
        "n_total": n_tot,
        "n_watch": n_watch,
        "n_step": n_step,
        "plot_fits": True,
        "dry_run": args.dry,
        "timestep_prior": args.partition_interval,
        "interest_index": args.index_interval,
    }

    prefix_str = "{0}_v_{1}_c_{2}_m_{3}_n_{4}_a_{5}_" "b_{6}_f_{7}_case_{8}".format(
        args.origin,
        args.version,
        data_cols,
        args.nsamples,
        n,
        low_f,
        hi_f,
        args.func,
        args.case,
    )

    logging.info("prefix str: {0}".format(prefix_str))
    kwargs_list = [
        {**barebone_dict_pars, **generate_fnames(prefix_str, j), **{"data_dict": d}}
        for j, d in zip(rr, dataset)
    ]

    super_kwargs_list = [
        kwargs_list[k :: args.nparallel] for k in range(args.nparallel)
    ]

    qu = mp.Queue()

    decorated = decorate_wlogger(
        function_dict[args.func], qu, join(args.logspath, args.logfilename)
    )

    processes = [
        mp.Process(target=decorated, args=(it, l_kw))
        for it, l_kw in zip(range(args.nparallel), super_kwargs_list)
    ]

    for p in processes:
        p.start()

    print("pro start")
    for p in processes:
        p.join()
    print("pro joined")

    # super_results_list = [qu.get() for p in processes]

    # results_list = [item for sublist in super_results_list for item in sublist]

    # reports_list = list(map(lambda x: x[0], results_list))

    logging.info("execution complete")
