from bm_support.posterior_tools import fit_step_model_d_v2
import gzip
import pickle
import logging
from os.path import join
logging.basicConfig(level=logging.INFO)

with gzip.open('../../../data/data_batches_identity_ai_hiai_pos_200.pgz') as fp:
    dataset = pickle.load(fp)


def generate_fnames(j):
    return {'figname_prefix': 'model_notau_{0}'.format(j),
            'tracename_prefix': 'trace_model_notau_{0}'.format(j)}

n_it = 45
dataset = dataset[n_it:n_it+1]
n_tot = 10000
n_watch = int(0.9*n_tot)
n_step = 10

barebone_dict_pars = {'n_features': 2,
                      'fig_path': './../../../figs/', 'trace_path': './../../../traces/',
                      'n_total': n_tot, 'n_watch': n_watch, 'n_step': n_step, 'plot_fits': True}


kwargs_list = [{**barebone_dict_pars, **generate_fnames(j), **{'data_dict': d}} for j, d in
               zip(range(len(dataset)), dataset)]

results_list = map(lambda kwargs: fit_step_model_d_v2(**kwargs), kwargs_list)

reports_list = list(map(lambda x: x[1], results_list))

with gzip.open(join('./../../../reports/',
                    '{0}.pgz'.format('report_identity_ai_hiai_pos_200')), 'wb') as fp:
    pickle.dump(reports_list, fp)


