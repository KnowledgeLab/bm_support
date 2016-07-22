from numpy import hstack, cumsum
from numpy.random import RandomState
from prob_constants import t0_min, t0_max, \
                        mu_min, mu_max, \
                        tau_min, tau_max


def generate_log_normal_mixture(n_modes=3, seed=123, n_samples=100,
                                t0_range=[t0_min, t0_max],
                                mu_range=[mu_min, mu_max],
                                tau_range=[tau_min, tau_max]):
    rns = RandomState(seed)

    ps = rns.dirichlet([5.]*n_modes)
    ns = rns.multinomial(n_samples, ps)

    mus_init = rns.uniform(mu_range[0], mu_range[1], n_modes)
    taus_init = rns.uniform(tau_range[0], tau_range[1], n_modes)
    t0s_init_prep = t0_range[0] + (t0_range[1] - t0_range[0])*rns.dirichlet([5.]*(n_modes+2))
    t0s_init = cumsum(t0s_init_prep)[:-2]
    values_list = [rns.lognormal(m, 1./s**0.5, size=n) + t0 for (m, s, t0, n) in
                   zip(mus_init, taus_init, t0s_init, ns)]
    data = hstack(values_list).astype(float)
    rns.shuffle(data)
    data = data[data < t0_range[1]]
    pps = zip(ps, mus_init, taus_init, t0s_init)

    return data, pps
