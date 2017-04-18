import time
import multiprocessing as mp
from numpy import array, arange, histogram, argmax, greater, nan, log
from numpy import sqrt, floor, ones, sum, exp, vstack, int, float32
from numpy import histogram
from numpy.random import RandomState
from bm_support.prob_constants import very_low_logp
from itertools import product
import theano.tensor as tt
from scipy import stats
from os.path import join
import logging
import gzip
import pickle
from scipy.signal import argrelextrema
from scipy.optimize import fmin_powell
from matplotlib import use
use('Agg')

from .math_aux import steplike_logistic, dist_hguess_hstate, dist_hguess_hstate_timestep
from .guess import guess_ranges, generate_beta_step_guess
from .param_converter import map_parameters
import pymc3 as pm
from pymc3 import find_MAP, DensityDist, Metropolis, sample, traceplot
from bm_support.math_aux import steplike_logistic, logp_bmix_shift_claims
from datahelpers.plotting import plot_beta_steps
from matplotlib.pyplot import close


def trim_data(data, n_bins=10):
    # find left and right bounds such that
    # the number of counts is above average (deviates from uniform)

    n = data.shape[0]
    cnts, xcoords = histogram(data, bins=n_bins)
    mean_counts = float(n) / n_bins
    cnts_flags = (cnts > mean_counts).astype(int)
    lb, rb = argmax(cnts_flags), argmax(cnts_flags[::-1])
    return xcoords[lb], xcoords[-(rb+1)]


def histogram_range(data, xr, n_bins=20):
    # bin data in n_bins within xr
    delta = (xr[1] - xr[0]) / n_bins
    bins = arange(xr[0], xr[1] + delta, delta)
    cnts, xcoords = histogram(data, bins)
    return cnts, xcoords


def find_max_neighbourhood(data, i):
    # return the size of the neighbourhood
    # for which data[i] is a local maximum
    # and the sum of data over the neighbourhood
    j = 0
    k = 0
    while i + k + 1 < data.shape[0] and data[i + k + 1] < data[i + k]:
        k += 1
    while i - j - 1 > 0 and data[i - j - 1] < data[i - j]:
        j += 1
    if k * j == 0:
        k = 0
        j = 0
    return 0.5 * (k + j), sum(data[i - j:i + k + 1]),


def analyse_local_maxima(data, xr=None, n_bins=20,
                         n_ext=2, alpha=0.5, beta=0.5,
                         gamma=0.5, logx=False):
    """

    :param data:
    :param xr:
    :param n_bins:
    :param n_ext:
    :param alpha: fraction
    :param beta:
    :param gamma:
    :return:
    """

    if logx:
        data = log(data.copy())
        if xr:
            xr = (log(xr[0]), log(xr[1]))

    if not xr:
        xr = min(data), max(data)

    cnts, xcoords = histogram_range(data, xr, n_bins)
    # the maxima are either a) in the interior or b) the boundaries
    # a) interior of cnts
    # get the extrema indices
    inds_ext = argrelextrema(cnts, greater)[0]
    # print type(inds_ext), inds_ext
    # print zip(xcoords, cnts)
    if len(inds_ext) > 0:
        # get the extrema values
        cnts_ext = cnts[inds_ext]
        # rank the extrema
        ranks_ext = cnts_ext.argsort()

        # take top n_ext extrema
        # provide top n_ext indices in reverse (descending) order
        # arg_sort is in ascending order
        top_inds = inds_ext[ranks_ext[:-n_ext - 1:-1]]
        fs = cnts[top_inds]
        nhoods = array([find_max_neighbourhood(cnts, pos) for pos in top_inds]).T
        # print nhoods
        report = True, 'Posterior acceptable'
        ans = xcoords[top_inds[0]]
        # is the second peak smaller than alpha * (first peak)?
        if top_inds.shape[0] > 1 and alpha * fs[0] < fs[1]:
            # is the sum over the second peak neighbourhood greater than
            # beta * sum over the first peak neighbourhood?
            if beta * nhoods[1, 0] < nhoods[1, 1]:
                # are the two candidate peaks distinct?
                # i.e. if they are close enough to each other, it might be the same peak
                if abs((xcoords[top_inds[0]] - xcoords[top_inds[1]])/(xr[1] - xr[0])) > gamma:
                    report = False, 'Posterior unacceptable: at least two prominent peaks'
                    # ans = nan
    else:
        report = False, 'Posterior unacceptable: extremum at the boundary'
        ans = xcoords[argmax(cnts)]
    if logx:
        ans = exp(ans)
    return ans, report


def analyse_flatness(data, xr=None, alpha=0.05):

    if not xr:
        xr = min(data), max(data)
    data_cut = data[(data >= xr[0]) & (data <= xr[1])]
    length = xr[1] - xr[0]
    # r[0] - ks-stat; r[1] - p-value
    r = stats.kstest(data_cut, stats.uniform(loc=xr[0], scale=length).cdf)
    if r[1] < alpha:
        report = True, 'Posterior is likely not flat'
    else:
        report = False, 'Posterior might be flat'
    return r[1], report


def fit_step_model(data_set, verbosity=0, plot_fits=False, fname_prefix='abc',
                   fpath='./', n_total=10000,
                   n_watch=9000, n_step=10):
    set_id = data_set[0]
    data = data_set[1]
    logging.info('Processing id', set_id,
                 'size of data set is', data.shape[1],
                 'freq is ', float(sum(data[-1]))/data.shape[1])

    n_features = data.shape[0] - 3
    n_features_ext = n_features + 1
    seed = 17
    beta_min, beta_max = -4., 4.
    gamma_min, gamma_max = 0.1, 1.
    slow, shi = -5., 5.0

    (tlow, thi), (mu_min, mu_max) = guess_ranges(data[0])

    with pm.Model(verbose=0) as model:
        #     x_priors = [pm.Beta('xprior_%d' % (i+1), alpha=10, beta=10) for i in range(n_features)]
        #     x_obss = [pm.Bernoulli('xobs_%d' % i, p=x_priors[i], observed=data[i+2]) for i
        #               in range(n_features)]

        beta_l = [pm.Normal('betaLeft_%d' % i, sd=2) for i in range(n_features_ext)]
        beta_r = [pm.Normal('betaRight_%d' % i, sd=2) for i in range(n_features_ext)]
        # beta_s = [pm.Normal('betaSteep_%d' % i, sd=1) for i in range(n_features_ext)]
        beta_s = [pm.Uniform('betaSteep_%d' % i, lower=slow, upper=shi) for i in range(n_features_ext)]

        beta_c = [pm.Uniform('betaCenter_%d' % i, lower=tlow, upper=thi)
                  for i in range(n_features_ext)]

        #     p_min_potential = pm.Potential('p_min_potential', tt.switch(tt.min(pi_) < .1, -np.inf, 0))
        order_means_potential = [pm.Potential('order_means_potential_%d' % j,
                                              tt.switch(beta_right > beta_left, 0, very_low_logp))
                                 for j, beta_left, beta_right in
                                 zip(range(n_features_ext), beta_l, beta_r)]

        data_xs = vstack([data[0], data[-1]])
        ys = DensityDist('yobs', steplike_logistic(beta_l, beta_r, beta_c, beta_s), observed=data)

    infos = []
    rns = RandomState(seed)

    for ite in range(20):
        seed = rns.randint(500)
        guess = {}
        #     guess_ln = {}
        #     guess.update(guess_ln)
        guess_betas = generate_beta_step_guess(data[0], n_features + 1, 'random', seed,
                                               (beta_min, beta_max),
                                               (gamma_min, gamma_max),
                                               names={'beta_c': 'betaCenter',
                                                      'beta_l': 'betaLeft',
                                                      'beta_r': 'betaRight',
                                                      'beta_s': 'betaSteep'})

        guess.update(guess_betas)

        #     guess_xprior = bg.generate_bernoulli_guess(data[1:-1], 'xprior', 0)
        #     guess.update(guess_xprior)

        sb_ranges = {
            'betaCenter': [tlow, thi],
            'betaSteep': [slow, shi],
        }

        model_dict = {
            'betaLeft': {'type': pm.Normal},
            'betaRight': {'type': pm.Normal},
            # 'betaSteep': {'type': pm.Normal},
            'betaSteep': {'type': pm.Uniform},
            'betaCenter': {'type': pm.Uniform},
            # 'xprior': {'type': pm.Beta}

        }

        fwd_gu = map_parameters(guess, model_dict, sb_ranges, True)

        with model:
            ll = model.logp(fwd_gu)

        dargs = {'xtol': 1e-07, 'fmin': fmin_powell}

        with model:
            best = find_MAP(start=fwd_gu, vars=model.vars, **dargs)

        raw_best_dict = map_parameters(best, model_dict, sb_ranges, False)
        #     rdcmp = bp.dict_cmp(raw_best_dict, pps)

        #     tot_error = sqrt(sum(array(rdcmp.values())**2))
        lp = model.logp(best)
        #         print('initial logp:', ll, 'iteration:', ite, 'logp =  ', lp)
        infos.append((lp, raw_best_dict, best))

    # sorted w.r.t to log likelihood
    sorted_first = list(infos)
    sorted_first.sort(key=lambda tup: tup[0], reverse=True)
    if verbosity > 0:
        logging.debug(sorted_first[0])

    raw_best_dict_plot_beta = {k: sorted_first[0][1][k]
                               for k in sorted_first[0][1].keys() if 'beta' in k and not 'xprior' in k}

    # raw_best_dict_plot_beta = {k : sorted_second[0][2][k] for k in pps if 'beta' in k and not 'xprior' in k}

    # pps_plot_beta = {k : pps[k] for k in pps if 'beta' in k and not 'xprior' in k}

    dict_base = {'betaCenter_': 't0', 'betaLeft_': 'b1', 'betaRight_': 'b2', 'betaSteep_': 'g'}

    fname_pre = fname_prefix + '_' + str(set_id)
    if plot_fits:
        plot_beta_steps(n_features_ext, dict_base, raw_best_dict_plot_beta, tlow, thi, sorted_first,
                        fname_prefix=fname_pre, path=fpath)

    with model:
        step = Metropolis()
        trace = sample(n_total, step, start=sorted_first[0][1], progressbar=False)

    varnames = [name for name in trace.varnames if not name.endswith('_')]
    report = {}
    traces = {}
    report['data_size'] = (data.shape[1], (True, 'Size of data'))
    report['freq'] = (float(sum(data[-1]))/data.shape[1], (True, 'Frequency of positives'))

    for k in varnames:
        traces[k] = trace[n_watch::n_step][k]
        #     print min(trace_cur), mean(trace_cur), median(trace_cur), max(trace_cur)
        a, b = trim_data(traces[k])
        # a = min(trace_cur)
        # b = max(trace_cur)
        report[k] = analyse_local_maxima(traces[k], (a, b), 10, n_ext=2)
        report['flat_' + k] = analyse_flatness(traces[k], (a, b))
        if verbosity > 0:
            logging.debug(k, report[k])

    bool_report = {k: report[k][1][0] for k in report.keys()}

    if not all(bool_report.values()):
        with model:
            axx = traceplot(trace[n_watch::n_step], fname_prefix=fname_pre, path=fpath)
            close()

    return report, traces


def fit_step_model_d(data_set, verbosity=0, plot_fits=False, fname_prefix='abc', fpath='./', n_total=10000,
                     n_watch=9000, n_step=10):
    set_id = data_set[0]
    fname_pre = fname_prefix + '_' + str(set_id)
    data = data_set[1]

    logging.info('Processing id', set_id,
                 ', size of data set is', data.shape[1],
                 ', freq is', float(sum(data[-1])) / data.shape[1])

    n_features = data.shape[0] - 3
    n_features_ext = n_features + 1
    n_modes = 2
    seed = 17
    beta_min, beta_max = -4., 4.
    gamma_min, gamma_max = 0.1, 1.
    slow, shi = -5., 5.0

    (tlow, thi), (mu_min, mu_max) = guess_ranges(data[0])

    with pm.Model(verbose=0) as model:

        beta_f = [pm.Normal('beta_%d' % i, sd=2) for i in range(n_features_ext)]

        t0_prior = pm.Uniform('t0', lower=tlow, upper=thi)
        pi_priors_left = pm.Dirichlet('piLeft', a=ones(n_modes), shape=(n_modes,))
        pi_priors_right = pm.Dirichlet('piRight', a=ones(n_modes), shape=(n_modes,))
        guess_potential = pm.Potential('guess_potential', tt.switch(beta_f[0] > 0.0, 0.0, very_low_logp))

        ys = DensityDist('yobs', logp_bmix_shift_claims(pi_priors_left,
                                                        pi_priors_right,
                                                        t0_prior, beta_f, n_modes), observed=data)
    infos = []
    rns = RandomState(seed)

    for ite in range(3):
        guess = {}
        guess.update({'piLeft': rns.dirichlet(n_modes * [1.])})
        guess.update({'piRight': rns.dirichlet(n_modes * [1.])})
        tmin = min(data[0])
        tmax = max(data[0])
        guess.update({'t0': rns.uniform(tmin, tmax)})
        guess.update({'beta_%d' % i: rns.normal(0.0, 1.0) for i in range(n_features_ext)})

        sb_ranges = {
            't0': (tmin, tmax)
        }

        model_dict = {
            'beta': {'type': pm.Normal},
            't0': {'type': pm.Uniform},
            'piLeft': {'type': pm.Dirichlet},
            'piRight': {'type': pm.Dirichlet}
        }

        fwd_gu = map_parameters(guess, model_dict, sb_ranges, True)

        with model:
            ll = model.logp(fwd_gu)

        dargs = {'xtol': 1e-07, 'fmin': fmin_powell}

        with model:
            best = find_MAP(start=fwd_gu, vars=model.vars, **dargs)

        raw_best_dict = map_parameters(best, model_dict, sb_ranges, False)

        lp = model.logp(best)
        infos.append((lp, raw_best_dict, best))

    # sorted w.r.t to log likelihood
    sorted_first = list(infos)
    sorted_first.sort(key=lambda tup: tup[0], reverse=True)
    if verbosity > 0:
        logging.debug(sorted_first[0])
        logging.debug(len(sorted_first))

    with model:
        step = Metropolis()
        trace = sample(n_total, step, start=sorted_first[0][1], progressbar=False)

    varnames = [name for name in trace.varnames if not name.endswith('_')]
    report = {}
    traces = {}
    report['data_size'] = (data.shape[1], (True, 'Size of data'))
    report['freq'] = (float(sum(data[-1])) / data.shape[1], (True, 'Frequency of positives'))
    report['range_t0'] = (min(data[0]), max(data[0]))
    report['set_id'] = set_id
    posterior_info = {}

    for k in varnames:
        traces[k] = trace[n_watch::n_step][k]
        arr = trace[k].copy()
        if len(arr.shape) > 1:
            arr = arr[:, 0]
        #     print min(trace_cur), mean(trace_cur), median(trace_cur), max(trace_cur)
        a, b = trim_data(arr)
        posterior_info['post_' + k] = analyse_local_maxima(arr, (a, b), 10, n_ext=2)
        posterior_info['flatness_' + k] = analyse_flatness(arr, (a, b))

    report['posterior_info'] = posterior_info

    bool_report = {k: posterior_info[k][1][0] for k in posterior_info.keys()}

    if not all(bool_report.values()):
        with model:
            axx = traceplot(trace[n_watch::n_step], fname_prefix=fname_pre, path=fpath)
            close()

    return report, traces


def fit_step_model_d_v2(data_dict, n_features, plot_fits=False,
                        figname_prefix='abc', fig_path='./',
                        tracename_prefix='tr', trace_path='./',
                        reportname_prefix='rep', report_path='./',
                        n_total=10000,
                        n_watch=9000, n_step=10, seed=17, n_map=1):
    n_modes = 2
    t0 = time.time()
    logging.info('Processing ids: {0}'.format(list(data_dict.keys())))
    logging.info('Sizes of data sets: {0}'.format(' '.join([str(x.shape[1]) for x in data_dict.values()])))
    shapes_dict = {k: (data_dict[k][-1].sum(), data_dict[k].shape[1]) for k in data_dict.keys()}
    posneg_dict = {k: array([shapes_dict[k][0], shapes_dict[k][1] - shapes_dict[k][0]]) for k in shapes_dict.keys()}

    with pm.Model() as model:
        beta_f = [pm.Normal('beta_{0}'.format(i + 1), mu=0, sd=4) for i in range(n_features)]
        beta_dummy = pm.Lognormal('beta0', mu=0, tau=0.25)
        pi_priors = {k: pm.Dirichlet('pi_{0}'.format(k), a=ones(n_modes)) for k in
                     data_dict.keys()}

        ys = DensityDist('yobs', dist_hguess_hstate(beta_dummy, beta_f, pi_priors), observed=data_dict)

    infos = []
    rns = RandomState(seed)

    for ite in range(n_map):
        guess = {}
        guess.update({'pi_{0}'.format(k): rns.dirichlet(0.1 * posneg_dict[k]) for k in posneg_dict})
        guess.update({'beta_{0}'.format(i + 1): rns.normal(0.0, 1.0) for i in range(n_features)})
        guess.update({'beta0': rns.lognormal(0.0, 1.0)})
        sb_ranges = {}

        model_dict = {
            'beta0': {'type': pm.Lognormal},
            'beta': {'type': pm.Normal},
            'pi': {'type': pm.Dirichlet}
            #             't0': {'type': pm.Uniform},
        }

        fwd_gu = map_parameters(guess, model_dict, sb_ranges, True)

        with model:
            ll = model.logp(fwd_gu)

        dargs = {'xtol': 1e-07, 'fmin': fmin_powell, 'disp': False}

        with model:
            best = find_MAP(start=fwd_gu, vars=model.vars, **dargs)

        raw_best_dict = map_parameters(best, model_dict, sb_ranges, False)

        lp = model.logp(best)
        infos.append((lp, raw_best_dict, best))

    # sorted w.r.t to log likelihood
    sorted_first = list(infos)
    sorted_first.sort(key=lambda tup: tup[0], reverse=True)
    logging.info('MAP estimates: {0} {1}'.format(len(sorted_first), sorted_first[0]))

    t1 = time.time()
    logging.info('map of batch {1} took {0:.2f} sec'.format(t1 - t0, list(data_dict.keys())))

    with model:
        step = Metropolis()
        trace = sample(n_total, step, init=None, start=sorted_first[0][1])

    t2 = time.time()

    varnames = [name for name in trace.varnames if not name.endswith('_')]
    report = {}
    traces = {}
    report['data_size'] = ({k: v.shape[1] for k, v in data_dict.items()}, (True, 'data sizes'))
    report['freq'] = ({k: float(sum(v[-1])) / v.shape[1] for k, v in data_dict.items()},
                      (True, 'Frequency of positives'))

    posterior_info = {'point': {}, 'flatness': {}}

    trs = {}

    for k in varnames:
        arr = trace[k].copy()
        trs[k] = arr
        arr = arr[n_watch::n_step]
        if len(arr.shape) > 1:
            arr = arr[:, 0]
        a, b = trim_data(arr)
        if k in model_dict.keys() and model_dict[k]['type'] == pm.Lognormal:
            flag = True
        else:
            flag = False
        posterior_info['point'][k] = analyse_local_maxima(arr, (a, b), n_bins=10, n_ext=2, logx=flag)
        posterior_info['flatness'][k] = analyse_flatness(arr, (a, b))

    report['posterior_info'] = posterior_info

    bool_report = [posterior_info['point'][k][1][0] for k in posterior_info['point'].keys()]
    bool_report.extend([posterior_info['flatness'][k][1][0] for k in posterior_info['flatness'].keys()])

    if not all(bool_report) or plot_fits:
        with model:
            axx = traceplot(trace[n_watch::n_step])
            fig = axx[0, 0].get_figure()
            fig.savefig(join(fig_path, '{0}.pdf'.format(figname_prefix)))
            close()

    with gzip.open(join(report_path, '{0}.pgz'.format(reportname_prefix)), 'wb') as fp:
        pickle.dump(report, fp)

    with gzip.open(join(trace_path, '{0}.pgz'.format(tracename_prefix)), 'wb') as fp:
        pickle.dump(trs, fp)

    logging.info('mcmc of batch {1} took {0:.2f} sec'.format(t2 - t1, list(data_dict.keys())))

    return model, report, trace, traces


def fit_model_f(data_dict, n_features, plot_fits=False,
                figname_prefix='abc', fig_path='./',
                tracename_prefix='tr', trace_path='./',
                reportname_prefix='rep', report_path='./',
                n_total=10000,
                n_watch=9000, n_step=10,
                seed=17, n_map=1,
                timestep_prior=array([4.0, 2.0]),
                dry_run=True):
    n_modes = 2
    n_times = 2

    logger = mp.get_logger()
    t0 = time.time()
    logger.info('Processing ids: {0}'.format(list(data_dict.keys())))
    logger.info('Sizes of data sets: {0}'.format(' '.join([str(x.shape[1]) for x in data_dict.values()])))
    shapes_dict = {k: (data_dict[k][-1].sum(), data_dict[k].shape[1]) for k in data_dict.keys()}
    posneg_dict = {k: array([shapes_dict[k][0], shapes_dict[k][1] - shapes_dict[k][0]]) for k in shapes_dict.keys()}
    intervals_enum = [chr(i) for i in range(ord('a'), ord('b') + 1)]
    if dry_run:
        logger.info('dry run ~ :p')
        model = None
        report = []
        trace = []
        traces = []
        t2 = time.time()
    else:
        with pm.Model() as model:
            beta_f = [pm.Normal('beta_{0}'.format(i + 1), mu=0, sd=4) for i in range(n_features)]
            beta_dummy = pm.Lognormal('beta0', mu=0, tau=0.25)
            pi_priors = {k:
                             [pm.Dirichlet('pi_{0}_{1}'.format(k, letter), a=ones(n_modes)) for letter in intervals_enum]
                         for k in data_dict.keys()}
            t0s = {k: pm.Dirichlet('t0_{0}'.format(k), a=ones(len(intervals_enum))) for k in
                   data_dict.keys()}
            ys = DensityDist('yobs', dist_hguess_hstate_timestep(beta_dummy, beta_f, pi_priors, t0s,
                                                                 n_times), observed=data_dict)

        infos = []
        rns = RandomState(seed)

        for ite in range(n_map):
            guess = {}
            guess.update({'pi_{0}_{1}'.format(k, l): rns.dirichlet(0.1 * posneg_dict[k])
                          for k, l in product(posneg_dict.keys(), intervals_enum)})
            guess.update({'beta_{0}'.format(i + 1): rns.normal(0.0, 1.0) for i in range(n_features)})
            guess.update({'beta0': rns.lognormal(0.0, 1.0)})
            guess.update({'t0_{0}'.format(k): rns.dirichlet(timestep_prior) for k in posneg_dict})
            sb_ranges = {}

            model_dict = {
                'beta0': {'type': pm.Lognormal},
                'beta': {'type': pm.Normal},
                'pi': {'type': pm.Dirichlet},
                't0': {'type': pm.Dirichlet},
            }

            fwd_gu = map_parameters(guess, model_dict, sb_ranges, True)

            with model:
                ll = model.logp(fwd_gu)

            dargs = {'xtol': 1e-07, 'fmin': fmin_powell, 'disp': False}

            with model:
                # best = find_MAP(start=fwd_gu, vars=model.vars, **dargs)
                best = find_MAP(start=fwd_gu, **dargs)

            raw_best_dict = map_parameters(best, model_dict, sb_ranges, False)

            lp = model.logp(best)
            infos.append((lp, raw_best_dict, best))

        # sorted w.r.t to log likelihood
        sorted_first = list(infos)
        sorted_first.sort(key=lambda tup: tup[0], reverse=True)
        logger.info('MAP estimates: {0} {1}'.format(len(sorted_first), sorted_first[0]))

        t1 = time.time()
        logger.info('map of batch {1} took {0:.2f} sec'.format(t1 - t0, list(data_dict.keys())))

        with model:
            step = Metropolis()
            trace = sample(n_total, step, init=None, start=sorted_first[0][1], progressbar=False)

        t2 = time.time()

        varnames = [name for name in trace.varnames if not name.endswith('_')]
        report = {}
        traces = {}
        report['data_size'] = ({k: v.shape[1] for k, v in data_dict.items()}, (True, 'data sizes'))
        report['freq'] = ({k: float(sum(v[-1])) / v.shape[1] for k, v in data_dict.items()},
                          (True, 'Frequency of positives'))
        posterior_info = {'point': {}, 'flatness': {}}

        trs = {}

        for k in varnames:
            arr = trace[k].copy()
            trs[k] = arr
            arr = arr[n_watch::n_step]
            if len(arr.shape) > 1:
                arr = arr[:, 0]
            a, b = trim_data(arr)
            if k in model_dict.keys() and model_dict[k]['type'] == pm.Lognormal:
                flag = True
            else:
                flag = False
            posterior_info['point'][k] = analyse_local_maxima(arr, (a, b), n_bins=10, n_ext=2, logx=flag)
            posterior_info['flatness'][k] = analyse_flatness(arr, (a, b))

        report['posterior_info'] = posterior_info

        bool_report = [posterior_info['point'][k][1][0] for k in posterior_info['point'].keys()]
        bool_report.extend([posterior_info['flatness'][k][1][0] for k in posterior_info['flatness'].keys()])

        if not all(bool_report) or plot_fits:
            with model:
                axx = traceplot(trace[n_watch::n_step])
                fig = axx[0, 0].get_figure()
                fig.savefig(join(fig_path, '{0}.pdf'.format(figname_prefix)))
                close()

        with gzip.open(join(report_path, '{0}.pgz'.format(reportname_prefix)), 'wb') as fp:
            pickle.dump(report, fp)

        with gzip.open(join(trace_path, '{0}.pgz'.format(tracename_prefix)), 'wb') as fp:
            pickle.dump(trs, fp)

        logger.info('mcmc of batch {1} took {0:.2f} sec'.format(t2 - t1, list(data_dict.keys())))
    logger.info('Calculation of batch id took {0:.2f} sec'.format(t2 - t0))

    return report, traces
