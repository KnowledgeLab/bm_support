from numpy import hstack, cumsum, sum, exp, log,\
    median, dot,\
    vstack, repeat, reshape, \
    ones, arange, array, append
from numpy.random import RandomState
from prob_constants import t0_min, t0_max, \
                        mu_min, mu_max, \
                        tau_min, tau_max, norm_const


def f_logistic(x):
    return 1 / (1 + exp(-x))


def logit(x):
    return log(x/(1. - x))


def generate_log_normal_mixture(n_modes=3, seed=123, n_samples=100,
                                t0_range=(t0_min, t0_max),
                                mu_range=(mu_min, mu_max),
                                tau_range=(tau_min, tau_max),
                                names={'ns': 'ns',
                                       't0': 't0', 'mu': 'mu',
                                       'tau': 'tau', 'ps': 'ps'}
                                ):
    # TODO pass state rather than seed to dependents

    """

    :param n_modes: number of log normal modes
    :param seed: random seed
    :param n_samples: total number of datapoints
    :param t0_range: range of t coordinates
    :param mu_range: range of mean parameters mu
    :param tau_range: range of precision parameters tau
    :param names: dict of
    :return:
    """
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
                                                     'beta': 'beta',
                                                     'prior_p': 'pp'}):
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

    xy_data, beta_pps, probs = \
        generate_logistic_y_from_bernoulli_x(n_features, pps[names['ns']], rns,
                                             names=names)

    data = vstack([reshape(values, (1, values.shape[0])), xy_data])

    rns.shuffle(data.T)

    # size = (n_features + 1, n_cycles)
    # pps = vstack([pps, betas])
    pps.update(beta_pps)
    data = data[:, data[0, :] < t0_range[1]]

    return data, pps


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


def generate_bernoullis(n_features=2, n_samples=30, p_range=(0.2, 0.8),
                        mode='determ', rns=None, seed=123):
    """
    generate n_features bernoulli parameters
        and for each generate n_samples datapoints

    :param n_features: number of features
    :param n_samples: number of datapoints
    :param p_range: range of bernoulli parameters
    :param mode: mode of bernoulli parameters generation
    :param rns: random state
    :param seed: seed for random generation, used only if no rns provided
    :return:
    NB: shape of x_data is (n_features+1, n_samples),
        one row of dummy variables is added
    """
    if n_features > 0:
        p1, p2 = p_range
        if not rns:
            rns = RandomState(seed)
        if mode == 'determ':
            x_bernoulli_p = arange(p1, p2, (p2 - p1) / n_features) + 0.5 * (p2 - p1) / n_features
        elif mode == 'random':
            x_bernoulli_p = rns.uniform(p1, p2, n_features)
        else:
            raise ValueError('mode parameter value \'{}\' is \
                             not one of : \'random\', \'determ\''.format(mode))
        x_bernoulli_p = append(x_bernoulli_p, [1.0])
        x_data = rns.binomial(1, repeat(reshape(x_bernoulli_p, (n_features + 1, 1)), n_samples, axis=1))
    else:
        x_bernoulli_p = array(1.)
        x_data = reshape(ones(n_samples), (1, n_samples))
    return x_data, x_bernoulli_p


def generate_betas(n_features=2, ns_list=(100, 200), beta_range=(-1, 1), rns=None, seed=123):
    """
    :param n_features: number of features
    :param ns_list: list of partition of the datapoints between cycles
    :param beta_range:
    :param rns: random state used to generate betas
    :param seed: seed for random generation, used only if no rns provided
    :return:
    """
    beta1, beta2 = beta_range
    n_cycles = len(ns_list)
    if not rns:
        rns = RandomState(seed)
    betas = rns.uniform(beta1, beta2, size=(n_features + 1, n_cycles))
    betas_np = hstack([repeat(betas.T[j], ns_list[j]).reshape(n_features + 1, ns_list[j]) for j in range(n_cycles)])
    return betas_np, betas


def convolve_logistic(x_data, betas_data, rns=None, seed=123):
    """

    :param x_data: data matrix of shape x_features, x_datapoints
    :param betas_data: coeff matrix of shape x_features, x_datapoints
    :param rns: random state used to generate categorical y
    :param seed: seed for random generation, used only if no rns provided
    :return:
    """
    probs = f_logistic(sum(betas_data * x_data, axis=0))
    if not rns:
        rns = RandomState(seed)
    y_data = rns.binomial(1, probs)
    return y_data, probs


def generate_logistic_y_from_bernoulli_x(n_features=3, ns_list=(100, 200), rns=None, seed=123,
                                         names={'beta': 'beta', 'prior_p': 'prior_p'}):

    """

    :param n_features:
    :param ns_list: list of partition of the datapoints between cycles
    :param rns: random state used to generate random quantities in sequence
    :param seed: seed for random generation, used only if no rns provided
    :param names:
    :return:
    """
    n_samples = sum(ns_list)
    if not rns:
        rns = RandomState(seed)
    x_data_ext, ps = generate_bernoullis(n_features, n_samples, (0.2, 0.8), 'determ', rns)
    betas_np, betas = generate_betas(n_features, ns_list, (-2., 2.), rns)
    y_data, logistic_probs = convolve_logistic(x_data_ext, betas_np, rns)

    xy_data = vstack([x_data_ext, y_data])
    pps = {names['beta'] + '_' + str(i): array(betas[:, i]) for i in range(betas.shape[1])}
    pps.update({names['prior_p'] + '_' + str(i): array(ps[i]) for i in range(n_features)})
    return xy_data, pps, logistic_probs















def generate_bernoulli_by_logistic_step(input_data, mode='determ', seed=123, n_features=1,
                                        beta_range=[-2, 2]):
    """
    given a 1D array of data, produce bernoulli random vars in a logistic manner

    :param input_data:
    :param mode:
    :param seed:
    :param beta_range:
    :return:
    """
    rns = RandomState(seed)

    if mode in ['random', 'determ']:
        if mode == 'determ':
            beta1, beta2 = -2, 2
            t0 = median(input_data)
        elif mode == 'random':
            betas = rns.uniform(beta_range, beta_range, size=(2, n_features))
            t0 = rns.uniform(min(input_data), max(input_data))
        gamma = 2
        data_logistic = array(map(lambda x: f_logistic(gamma * (x - t0)), input_data))

        betas = beta1 + (beta2 - beta1)*data_logistic
        mus = map(f_logistic, betas)
        out_data = rns.binomial(1, mus)
        return out_data
    else:
        raise ValueError('mode parameter value \'{}\' is \
                         not one of : \'random\', \'determ\''.format(mode))




def generate_logistic(n_features=3, ns_list=(100, 200), seed=123,
                      names={'beta': 'beta'}
                      ):
    rns = RandomState(seed)
    n_samples = sum(ns_list)
    n_cycles = len(ns_list)

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
