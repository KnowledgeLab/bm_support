from numpy import exp, sqrt, log, pi, ceil, dot
from numpy import concatenate, cumsum, cumprod
from numpy import ones, arange
from pymc3.math import logsumexp
import theano.tensor as tt
from prob_constants import very_low_logp
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def find_intlike_delta(a, b, n):
    if a < b:
        xs = float(b - a)/n
        if xs > 1:
            return ceil(xs)
        else:
            return 1./ceil(1./xs)
    else:
        raise ValueError('~(a < b)')


def inv_logit(p):
    return exp(p) / (1 + exp(p))


def logit(p):
    return log(p/(1-p))


def norm(x):
    return exp(-x**2/2)/sqrt(2*pi)


def unorm(x, m, s):
    return norm((x-m)/s)/s


def lnormal_shifted(x, m, s, t):
    return 0.0 if x <= t else unorm(log(x-t), m, s)/(x-t)

logodds_forward = logit
logistic = inv_logit
logodds_backward = inv_logit

def sb_forward(x):
    x0 = x[:-1]
    s = cumsum(x0[::-1], 0)[::-1] + x[-1]
    z = x0/s
    Km1 = x.shape[0] - 1
    k = arange(Km1)[(slice(None), ) + (None, ) * (x.ndim - 1)]
    eq_share = - log(Km1 - k)
    y = logit(z) - eq_share
    return y


def sb_backward(y):
    Km1 = y.shape[0]
    k = arange(Km1)[(slice(None), ) + (None, ) * (y.ndim - 1)]
    eq_share = - log(Km1 - k)
    z = inv_logit(y + eq_share)
    yl = concatenate([z, ones(y[:1].shape)])
    yu = concatenate([ones(y[:1].shape), 1-z])
    S = cumprod(yu, 0)
    x = S * yl
    return x


def log_forward(x):
    return log(x)


def log_backward(x):
    return exp(x)


def int_backward(a, b, x):
    r = (b - a) * exp(x) / (1 + exp(x)) + a
    return r


def int_forward(a, b, x):
    r = log((x - a) / (b - x))
    return r


def logp_ln_shifted_(mu, tau, t0, value):
    delta = lambda x: tt.log(x - t0) - mu
    return tt.switch(tt.gt(value, t0),
                     -0.5*(tt.log(2 * pi) + 2.0*tt.log(value - t0) - tt.log(tau)
                     + delta(value).dot(tau)*delta(value)),
                     very_low_logp)


def ln_shifted(mu, tau, t0):

    def logp_(value):
        def delta(x):
            return tt.log(x - t0) - mu

        return tt.switch(tt.gt(value, t0),
                         -0.5 * (tt.log(2 * pi) + 2.0 * tt.log(value - t0) - tt.log(tau)
                         + delta(value).dot(tau) * delta(value)), very_low_logp)

    return logp_


def logp_shln_steplike_logistic(beta_l, beta_r, beta_c, beta_s, mu, tau, t0):
    def logp_(value):
        # n_f x n_d
        betas = tt.stacklists([tt_logistic_step(b1, b2, c, gamma, value[0])
                               for (b1, b2, c, gamma) in
                               zip(beta_l, beta_r, beta_c, beta_s)])
        # n_f x n_d
        # xs = tt.stacklists([value[j + 1] for j in range(len(beta_l))])

        # 1 x n_d
        args = tt.sum(betas * value[1:-1], axis=0)
        # probability from logistic
        # 1 x n_d
        pr_log = tt_logistic(args)
        ll = tt.sum(value[-1] * tt.log(pr_log) +
                    (1. - value[-1]) * tt.log(1. - pr_log) +
                    logp_ln_shifted_(mu, tau, t0, value[0]))

        return ll

    return logp_


def logp_shln_steplike_logistic2(beta_l, beta_r, beta_c, beta_s, mu, tau, t0):
    def logp_(value):
        # n_f x n_d
        betas = tt.stacklists([tt_logistic_step(b1, b2, c, gamma, value[0])
                               for (b1, b2, c, gamma) in
                               zip(beta_l, beta_r, beta_c, beta_s)])

        penalty = tt.sum(tt.sum(tt.abs_(tt.stacklists([beta_l, beta_r]))))
        # 1 x n_d
        args = tt.sum(betas * value[1:-1], axis=0)

        # 1 x n_d : probability from logistic
        pr_log = tt_logistic(args)

        # n_f x n_d
        # xss = [value[j+1] for j in range(len(xp))]
        # 1 x n_d
        # xpp = tt.sum(tt.stacklists([v * tt.log(nu) + (1. - v)*tt.log(1. - nu) for v, nu in zip(xss, xp)]))

        ll = tt.sum(value[-1] * tt.log(pr_log) +
                    (1. - value[-1]) * tt.log(1. - pr_log) +
                    logp_ln_shifted_(mu, tau, t0, value[0])
                    # + xpp
                    )

        return ll + penalty

    return logp_


def steplike_logistic(beta_l, beta_r, beta_c, beta_s):
    def logp_(value):
        # n_f x n_d
        betas = tt.stacklists([tt_logistic_step(b1, b2, c, gamma, value[0])
                               for (b1, b2, c, gamma) in
                               zip(beta_l, beta_r, beta_c, beta_s)])

        penalty = tt.sum(tt.sum(tt.abs_(tt.stacklists([beta_l, beta_r]))))
        # 1 x n_d
        args = tt.sum(betas * value[1:-1], axis=0)

        # 1 x n_d : probability from logistic
        pr_log = tt_logistic(args)

        ll = tt.sum(value[-1] * tt.log(pr_log) +
                    (1. - value[-1]) * tt.log(1. - pr_log))

        return ll + penalty

    return logp_


def logp_shifted_ln_mix(pi, mus, taus, t0s):

    """
    Log likelihood of log-normal mixture distribution

    :param pi:
    :param mus:
    :param taus:
    :param t0s:
    :return:
    """

    def logp_(value):
        logps = [tt.log(pi[i]) + logp_ln_shifted_(mu, tau, t0, value)
                 for (i, mu, tau, t0) in zip(range(len(mus)), mus, taus, t0s)]

        return tt.sum(logsumexp(tt.stacklists(logps)[:, :], axis=0))

    return logp_


def logp_mixture(pis, func, **kwargs):
    """
    general mixture of funcs
    :param pis: Dirichlet distribution, class probs
    :param func: func (a, b,c, value)
    :param kwargs: dictionary {a: [a_1, a_2, ...], ...}
    :return:
    """

    def logp_(value):
        kw = dict(kwargs)
        n_dim = pis.tag.test_value.shape[0]
        kw['value'] = [value]*n_dim
        dds = [{key: kw[key][k] for key in kw.keys()} for k in range(n_dim)]
        logps = [tt.log(pis[i]) + func(**kw2)
                 for (i, kw2) in zip(range(n_dim), dds)]

        return tt.sum(logsumexp(tt.stacklists(logps)[:, :], axis=0))

    return logp_


# def logistic_step


def tt_inv_logit(arg):
    def logp_(value):
        return 1. / (1. + tt.exp(-value.dot(arg)))
    return logp_


def tt_logistic_step(b1, b2, t0, g, value):
    return b1 + (b2 - b1) / (1. + tt.exp(-g*(value - t0)))


def tt_logistic_step_dist(b1, b2, t0, g):
    def logp_(value):
        return b1 + (b2 - b1) / (1. + tt.exp(-g*(value - t0)))
    return logp_


def np_logistic_step(b1, b2, t0, g, value):
    return b1 + (b2 - b1) / (1. + exp(-g*(value - t0)))


def tt_logistic(value):
    return 1./(1. + tt.exp(-value))


def get_scalers(arr):
    # reshape (convert 1D array to 2D) is necessary for future compatability
    scalers_dict = {k: MinMaxScaler().fit(arr[k].reshape(-1, 1))
                    for k in range(arr.shape[0]) if len(set(arr[k])) > 2}
    return scalers_dict


def use_scalers(arr, scalers_dict):
    arr2 = arr.copy()
    for k in scalers_dict.keys():
        arr2[k] = scalers_dict[k].transform(arr[k].reshape(-1, 1)).flatten()
    return arr2
