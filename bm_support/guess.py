from numpy import log, exp, mean, min, max, std, array
from sklearn.cluster import KMeans
from numpy.random import RandomState
from prob_constants import norm_const
from prob_constants import t0_min, t0_max, \
                        mu_min, mu_max, \
                        tau_min, tau_max


def generate_log_normal_guess(data, n_modes=2, mode='random', seed=123):

    rns = RandomState(seed)
    tmin = min(data)
    tmax = max(data)

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
        taus_seed = array(sigmas_seed)**(-2)

        t0s_seed = [x-y[1] for x, y in zip(t0s_seed0, moments)]

    # t0s_seed = t0s_seed0
    # t0s_seed = [min(x) for x in datas]

    if mode == 't_uniform':
        t0s_seed = [k*(tmax-tmin)/n_modes + tmin + norm_const for k in range(n_modes)]

    if mode == 'random':
        t0s_seed = [rns.uniform(tmin, tmax) for i in range(n_modes)]
        mus_seed = [rns.uniform(mu_min, mu_max) for i in range(n_modes)]
        taus_seed = [rns.uniform(tau_min, tau_max) for i in range(n_modes)]
        ratios_seed = rns.dirichlet(n_modes*[5.])

    return ratios_seed, t0s_seed, mus_seed, taus_seed