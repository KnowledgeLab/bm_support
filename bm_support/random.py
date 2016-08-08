from numpy import hstack, cumsum, sum, exp, dot,\
    vstack, repeat, reshape, \
    ones, arange, array, append
from numpy.random import RandomState
from prob_constants import t0_min, t0_max, \
                        mu_min, mu_max, \
                        tau_min, tau_max, norm_const


def generate_log_normal_mixture(n_modes=3, seed=123, n_samples=100,
                                t0_range=(t0_min, t0_max),
                                mu_range=(mu_min, mu_max),
                                tau_range=(tau_min, tau_max),
                                names={'ns': 'ns',
                                       't0': 't0', 'mu': 'mu',
                                       'tau': 'tau', 'ps': 'ps'}
                                ):

    rns = RandomState(seed)
    data, pps = generate_log_normal(n_modes, seed, n_samples,
                                    t0_range, mu_range, tau_range,
                                    names)
    rns.shuffle(data)
    data = data[data < t0_range[1]]

    return data, pps


def generate_log_normal_mixture_with_logistic(n_modes=3, seed=123, n_samples=100,
                                              n_features=1,
                                              t0_range=(t0_min, t0_max),
                                              mu_range=(mu_min, mu_max),
                                              tau_range=(tau_min, tau_max),
                                              names={'ns': 'ns',
                                                     't0': 't0', 'mu': 'mu',
                                                     'tau': 'tau', 'ps': 'ps',
                                                     'beta': 'beta'}):
    """

    :param n_modes:
    :param seed:
    :param n_samples:
    :param n_features:
    :param t0_range:
    :param mu_range:
    :param tau_range:
    :param names
    :return:
    """

    rns = RandomState(seed)
    values, pps = generate_log_normal(n_modes, seed, n_samples,
                                      t0_range, mu_range, tau_range, names)

    # ns = array(pps[], dtype=int)
    xy_data, beta_pps, x_berns, probs = generate_logistic(n_features, pps[names['ns']],
                                                          n_modes, seed, names)

    data = vstack([reshape(values, (1, values.shape[0])), xy_data])

    rns.shuffle(data.T)

    # size = (n_features + 1, n_cycles)
    # pps = vstack([pps, betas])
    pps.update(beta_pps)
    data = data[:, data[0, :] < t0_range[1]]

    return data, pps, x_berns


def generate_log_normal(n_modes=3, seed=123, n_samples=100,
                        t0_range=(t0_min, t0_max),
                        mu_range=(mu_min, mu_max),
                        tau_range=(tau_min, tau_max),
                        names={'ns': 'ns',
                               't0': 't0', 'mu': 'mu',
                               'tau': 'tau', 'ps': 'ps'}
                        ):

    #TODO issue -- mutable default parameter names

    rns = RandomState(seed)

    ps = rns.dirichlet([5.]*n_modes)
    ns = rns.multinomial(n_samples, ps)

    collapse_coeff = 0.2

    mus_init = 0.5 * (mu_range[1] + mu_range[0]) + (mu_range[1] - mu_range[0]) * \
                                                   (collapse_coeff * (rns.uniform(0., 1.0, n_modes) - 0.5))

    taus_init = (tau_range[1] * tau_range[0]) ** 0.5 * (tau_range[1] / tau_range[0]) ** \
                                                       (collapse_coeff * (rns.uniform(0., 1.0, n_modes) - 0.5))

    # for n modes, we break the stick in n+1 places, and drop the left most piece
    # because the support of log-normal is half axis
    t0_ps = rns.dirichlet([5.] * (n_modes+2))
    t0s_init = (t0_range[0] + norm_const) + (t0_range[1] - t0_range[0] - norm_const) * cumsum(t0_ps[:-2])

    values = hstack([rns.lognormal(m, 1./s**0.5, size=n) + t0 for (m, s, t0, n) in
                     zip(mus_init, taus_init, t0s_init, ns)])

    # pps = array([ns, t0s_init,  mus_init, taus_init])
    pps_dict = {}
    pps_dict.update({names['t0'] + '_' + str(i): array(v) for i, v in zip(range(len(t0s_init)), t0s_init)})
    pps_dict.update({names['mu'] + '_' + str(i): array(v) for i, v in zip(range(len(mus_init)), mus_init)})
    pps_dict.update({names['tau'] + '_' + str(i): array(v) for i, v in zip(range(len(taus_init)), taus_init)})
    pps_dict['ns'] = array(ns, dtype=int)
    pps_dict[names['ps']] = array(ps)

    return values, pps_dict


def generate_logistic(n_features=3, ns_list=(100, 200),
                      n_cycles=2, seed=123,
                      names={'beta': 'beta'}
                      ):
    rns = RandomState(seed)
    n_samples = sum(ns_list)

    def f_logistic(x):
        return 1 / (1 + exp(-x))

    p1, p2 = 0.2, 0.8

    x_bernoulli_p = arange(p1, p2, (p2 - p1) / n_features) + 0.5 * (p2 - p1) / n_features
    x_data = rns.binomial(1, repeat(reshape(x_bernoulli_p, (n_features, 1)), n_samples, axis=1))
    x_data_ext = vstack([x_data, ones(n_samples)])

    beta1 = -1
    beta2 = 1
    betas = rns.uniform(beta1, beta2, size=(n_features + 1, n_cycles))
    betas_np = hstack([repeat(betas.T[j], ns_list[j]).reshape(n_features + 1, ns_list[j]) for j in range(n_cycles)])
    probs = f_logistic(sum(betas_np * x_data_ext, axis=0))
    y_data = rns.binomial(1, probs)
    xy_data = vstack([x_data_ext, y_data])
    pps = {names['beta'] + '_' + str(i): array(betas[:, i]) for i in range(betas.shape[1])}

    return xy_data, pps, x_bernoulli_p, probs
