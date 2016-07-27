from numpy import hstack, cumsum, sum, exp, dot,\
    vstack, repeat, reshape, \
    ones, arange, array
from numpy.random import RandomState
from prob_constants import t0_min, t0_max, \
                        mu_min, mu_max, \
                        tau_min, tau_max


def generate_log_normal_mixture(n_modes=3, seed=123, n_samples=100,
                                t0_range=(t0_min, t0_max),
                                mu_range=(mu_min, mu_max),
                                tau_range=(tau_min, tau_max)):

    rns = RandomState(seed)
    values_list, pps = generate_log_normal_list(n_modes, seed, n_samples,
                                                t0_range, mu_range, tau_range)
    data = hstack(values_list).astype(float)
    rns.shuffle(data)
    data = data[data < t0_range[1]]

    return data, pps


def generate_log_normal_mixture_with_logisitic(n_modes=3, seed=123, n_samples=100,
                                              t0_range=(t0_min, t0_max),
                                              mu_range=(mu_min, mu_max),
                                              tau_range=(tau_min, tau_max)):

    rns = RandomState(seed)
    values_list, pps = generate_log_normal_list(n_modes, seed, n_samples,
                                                t0_range, mu_range, tau_range)

    ns = array(pps, dtype=int)[:, 0]
    cat_list, y, z = generate_logistic(1, ns, n_modes, seed)

    data = hstack([vstack(pp) for pp in zip(values_list, cat_list)])

    rns.shuffle(data.T)
    data = data[:, data[0, :] < t0_range[1]]

    return data, pps


def generate_log_normal_list(n_modes=3, seed=123, n_samples=100,
                             t0_range=(t0_min, t0_max),
                             mu_range=(mu_min, mu_max),
                             tau_range=(tau_min, tau_max)):
    rns = RandomState(seed)

    ps = rns.dirichlet([5.]*n_modes)
    ns = rns.multinomial(n_samples, ps)

    collapse_coeff = 0.2

    mus_init = 0.5 * (mu_range[1] + mu_range[0]) + (mu_range[1] - mu_range[0]) * \
                                                   (collapse_coeff * (rns.uniform(0., 1.0, n_modes) - 0.5))

    taus_init = (tau_range[1] * tau_range[0]) ** 0.5 * (tau_range[1] / tau_range[0]) ** \
                                                       (collapse_coeff * (rns.uniform(0., 1.0, n_modes) - 0.5))
    t0s_init_prep = t0_range[0] + (t0_range[1] - t0_range[0])*rns.dirichlet([5.]*(n_modes+2))
    t0s_init = cumsum(t0s_init_prep)[:-2]
    values_list = [rns.lognormal(m, 1./s**0.5, size=n) + t0 for (m, s, t0, n) in
                   zip(mus_init, taus_init, t0s_init, ns)]
    pps = zip(ns, mus_init, taus_init, t0s_init)

    return values_list, pps


def generate_logistic(n_features=3, ns_list=(100, 200),
                      n_cycles=2, seed=123):
    rns = RandomState(seed)
    n_samples = sum(ns_list)

    def f_logistic(x):
        return 1 / (1 + exp(-x))

    p1, p2 = 0.2, 0.8
    categorical_ps = arange(p1, p2, (p2 - p1) / n_features) + 0.5 * (p2 - p1) / n_features
    data_features = rns.binomial(1, repeat(reshape(categorical_ps, (n_features, 1)),
                                           n_samples, axis=1))
    data_features_ext = vstack([data_features, ones(n_samples)])

    jjs = list(cumsum(ns_list))
    js = [0] + jjs[:-1]

    beta1, beta2 = -2, 2
    betas = [rns.uniform(beta1, beta2, size=n_features + 1) for i in range(n_cycles)]
    probs = [f_logistic(dot(beta, data_features_ext[:, j:jj])) for beta, j, jj in zip(betas, js, jjs)]
    categorical_data = [rns.binomial(1, prob) for prob in probs]
    return categorical_data, betas, probs


