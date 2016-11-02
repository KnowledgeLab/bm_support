from numpy import log, exp, mean, min, max, std, array, \
    floor, append, cumsum, vstack
from sklearn.cluster import KMeans
from numpy.random import RandomState
from prob_constants import norm_const
from prob_constants import tau_min, tau_max


def generate_log_normal_guess(data, n_modes=2, mode='random', seed=123,
                              names={'t0': 't0', 'mu': 'mu',
                                     'tau': 'tau', 'ps': 'ps'}):

    rns = RandomState(seed)
    tmin = min(data)

    (tlow, thi), (mu_min, mu_max) = guess_ranges(data)

    contracted_mu_min = 0.5*(mu_min + mu_max) - 0.25*(mu_max - mu_min)
    contracted_mu_max = 0.5*(mu_min + mu_max) + 0.25*(mu_max - mu_min)
    contracted_tau_min = (tau_min*tau_max)**0.5 * (tau_max/tau_min)**0.25
    contracted_tau_max = (tau_min*tau_max)**0.5 / (tau_max/tau_min)**0.25

    if mode == 'kmeans':
        m = KMeans(n_clusters=n_modes)
        data2 = log(data - min(data) + norm_const)
        m.fit(data2.reshape(-1, 1))

        t0s_seed0 = exp(m.cluster_centers_.flatten()) - norm_const + min(data)
        datas = [data[m.labels_ == k] for k in range(n_modes)]

        t0s_seed0, datas = zip(*sorted(zip(t0s_seed0, datas)))
        ratios_seed = [float(x.shape[0])/data.shape[0] for x in datas]

        moments = [(mean(t) - tmin, std(t)) for t in datas]

        mus_seed = [log(ms[0]/(1. + (ms[1]/ms[0])**2)**0.5) for ms in moments]
        sigmas_seed = [(log((1. + (ms[1]/ms[0])**2)))**0.5 for ms in moments]
        taus_seed = list(array(sigmas_seed)**(-2))

        t0s_seed = [x-y[1] for x, y in zip(t0s_seed0, moments)]

    elif mode in ['random', 't_uniform']:
        mus_seed = [rns.uniform(contracted_mu_min, contracted_mu_max) for i in range(n_modes)]
        taus_seed = [rns.uniform(contracted_tau_min, contracted_tau_max) for i in range(n_modes)]
        ratios_seed = list(rns.dirichlet(n_modes * [5.]))

        if mode == 't_uniform':
            t0s_seed = [k*(thi-tlow)/n_modes + tlow + norm_const for k in range(n_modes)]
        elif mode == 'random':
            ps = rns.dirichlet([5.]*n_modes)
            right = tlow + norm_const + (thi - tlow - norm_const)*cumsum(ps[:-1])
            t0s_seed = append(tlow + norm_const, right)
    else:
        mode_values = ['random', 't_uniform', 'kmeans']
        raise ValueError('keyword mode value should be either of: '
                         + ('{} ' * len(mode_values)).format(*mode_values))

    pps_dict = {}
    pps_dict.update({names['t0'] + '_' + str(i): array(v) for i, v in zip(range(len(t0s_seed)), t0s_seed)})
    pps_dict.update({names['mu'] + '_' + str(i): array(v) for i, v in zip(range(len(mus_seed)), mus_seed)})
    pps_dict.update({names['tau'] + '_' + str(i): array(v) for i, v in zip(range(len(taus_seed)), taus_seed)})
    if n_modes > 1:
        pps_dict[names['ps']] = array(ratios_seed)
    return pps_dict


def guess_ranges(data):
    tmin = min(data)
    tmax = max(data)
    tlen = tmax - tmin
    tlow = tmin - 0.25 * tlen
    thi = tmax + 0.25 * tlen

    mu_order = floor(log(0.1*tlen))
    mu_min = mu_order - 4.0
    mu_max = mu_order + 4.0
    return (tlow, thi), (mu_min, mu_max)


def generate_beta_step_guess(data, n_features, mode='random', seed=123,
                             beta_range=(-3., 3.),
                             gamma_range=(0., 10.),
                             names={'beta_c': 'beta_c',
                                    'beta_l': 'beta_l',
                                    'beta_r': 'beta_r',
                                    'beta_s': 'beta_s'}):
    rns = RandomState(seed)
    trange = min(data), max(data)
    #   beta centers
    beta_cs = rns.uniform(*trange, size=n_features)
    beta_range_ext = beta_range[0], mean(beta_range), beta_range[1]
    if mode == 'random':
        betas = rns.uniform(*beta_range_ext[::2], size=(2, n_features))
    else:
        betas = vstack([rns.uniform(*beta_range_ext[:-1], size=(1, n_features)),
                        rns.uniform(*beta_range_ext[1:], size=(1, n_features))])

    # beta steepness

    beta_ss = rns.uniform(*gamma_range, size=n_features)
    pps_dict = {}

    pps_dict.update({names['beta_s'] + '_' + str(i): array(v)
                     for i, v in zip(range(len(beta_ss)), beta_ss)})
    pps_dict.update({names['beta_c'] + '_' + str(i): array(v)
                     for i, v in zip(range(len(beta_cs)), beta_cs)})
    pps_dict.update({names['beta_l'] + '_' + str(i): array(v)
                     for i, v in zip(range(betas.shape[1]), betas[0])})
    pps_dict.update({names['beta_r'] + '_' + str(i): array(v)
                     for i, v in zip(range(betas.shape[1]), betas[1])})

    return pps_dict


def generate_bernoulli_guess(data, name, first_index):
    rdict = {name + '_%d' % (i+first_index):
             (1.- 1e-4)*float(sum(data[i]))/data.shape[1] for i in range(data.shape[0])}
    return rdict