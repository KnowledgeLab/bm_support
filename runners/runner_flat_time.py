import argparse
from bm_support.posterior_tools import fit_step_model_d_v2
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
    return {'figname_prefix': 'model_{0}_{1}_{2}'.format(modeltype, batch_size, j),
            'tracename_prefix': 'trace_{0}_{1}_{2}'.format(modeltype, batch_size, j)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--batchsize',
                        default='200',
                        help='Size of data batches')
    parser.add_argument('-b', '--begin',
                        default='0',
                        help='begin (starting) index in the batch')
    parser.add_argument('-e', '--end',
                        default='-1',
                        help='end (last) index in the batch')
    parser.add_argument('-m', '--modeltype',
                        default='identity_ai_hiai_pos',
                        help='model type specified by')
    parser.add_argument('-n', '--numberdraws',
                        default='1000',
                        help='mcmc number of draws')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if is_int(args.batchsize):
        batchsize = int(args.batchsize)
    else:
        raise ValueError('batchsize not an integer')

    if is_int(args.begin):
        begin = int(args.begin)
    else:
        raise ValueError('begin not an integer')

    if is_int(args.end):
        end = int(args.end)
    else:
        raise ValueError('end not an integer')

    if is_int(args.numberdraws):
        numberdraws = int(args.numberdraws)
    else:
        raise ValueError('numberdraws not an integer')
    modeltype = args.modeltype

    logging.info('batchsize : {0}'.format(batchsize))
    logging.info('begin : {0}'.format(begin))
    logging.info('end : {0}'.format(end))
    logging.info('numberdraws : {0}'.format(numberdraws))
    logging.info('modeltype : {0}'.format(modeltype))

    with gzip.open('../../../data/data_batches_identity_ai_hiai_pos_1000.pgz') as fp:
        dataset = pickle.load(fp)

    dataset = dataset[begin:end]
    n_tot = numberdraws
    n_watch = int(0.9*n_tot)
    n_step = 10

    barebone_dict_pars = {'n_features': 2,
                          'fig_path': './../../../figs/', 'trace_path': './../../../traces/',
                          'n_total': n_tot, 'n_watch': n_watch, 'n_step': n_step, 'plot_fits': True}

    kwargs_list = [{**barebone_dict_pars, **generate_fnames(modeltype, batchsize, j),
                    **{'data_dict': d}} for j, d in
                   zip(range(begin, end), dataset)]

    results_list = map(lambda kwargs: fit_step_model_d_v2(**kwargs), kwargs_list)

    reports_list = list(map(lambda x: x[1], results_list))

    with gzip.open(join('./../../../reports/',
                        'report_{0}_{1}.pgz'.format(modeltype, batchsize)), 'wb') as fp:
        pickle.dump(reports_list, fp)


