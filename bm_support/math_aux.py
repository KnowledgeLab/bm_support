from numpy import exp, sqrt, log, pi
from numpy import concatenate, cumsum, cumprod
from numpy import ones, arange
from pymc3.math import logsumexp
import theano.tensor as tt
from prob_constants import very_low_logp


def inv_logit(p):
    return exp(p) / (1 + exp(p))


def logit(p):
    return log(p/(1-p))


def norm(x):
    return exp(-x**2/2)/sqrt(2*pi)


def unorm(x, m, s):
    return norm((x-m)/s)/s


def ln_shifted(x, m, s, t):
    return 0.0 if x <= t else unorm(log(x-t), m, s)/(x-t)


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


def int_backward(a, b, x):
    r = (b - a) * exp(x) / (1 + exp(x)) + a
    return r


def int_forward(a, b, x):
    r = log((x - a) / (b - x))
    return r


def logp_ln_shifted(mu, tau, t0, value):
    delta = lambda x: tt.log(x - t0) - mu
    return tt.switch(tt.gt(value, t0),
                     -0.5*(tt.log(2 * pi) + 2.0*tt.log(value - t0) - tt.log(tau)
                     + delta(value).dot(tau)*delta(value)),
                     very_low_logp)


def tt_inv_logit(betas):
    def logp_(value):
        return 1. / (1. + tt.exp(-value.dot(betas)))
    return logp_

# Log likelihood of Gaussian mixture distribution
def logp_glmix(pi, mus, taus, t0s):
    def logp_(value):
        logps = [tt.log(pi[i]) + logp_ln_shifted(mu, tau, t0, value)
                 for (i, mu, tau, t0) in zip(range(len(mus)), mus, taus, t0s)]

        return tt.sum(logsumexp(tt.stacklists(logps)[:, :], axis=0))

    return logp_
