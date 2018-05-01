from numpy import array, sqrt, log, exp, pi, histogram, diag, abs, max
from scipy.optimize import curve_fit
from functools import partial
# this package find thi


def lnormal_off(x, mu, sigma, a, t0):
    """pdf of lognormal distribution"""
    xmin = x.min()
    if x.min() > t0:
        x = (x - t0).copy()
    else:
        x = array(1e-10*(t0 - xmin)*array([1]*x.shape[0]))
    return a*(exp(-(log(x) - mu)**2 / (2 * sigma**2)) / (x * sigma * sqrt(2 * pi)))


def fit_func(data, mode='pile', nbins=10, foo=lnormal_off, guess=(2., 1., 50, 0.5), return_error=False, verbose=False):
    """

    :param data:
    :param mode:
    :param nbins:
    :param foo:
    :param guess:
    :param verbose:
    :return:
    """
    if mode == 'pile':
        y, bnds = histogram(data, bins=nbins, normed=False)
        delta = bnds[1] - bnds[0]
        x = bnds[1:] - 0.5 * delta
        y = y / delta
    elif mode == 'hist':
        x, y = data
    else:
        return None

    guess_ = list(guess)
    guess_[2] = sum(y)
    popt, pcov = curve_fit(foo, x, y, p0=guess_)
    perr = sqrt(diag(pcov))
    yf = foo(x, *popt)
    terr = (sum((yf-y)**2)**0.5)/len(x)
    if verbose:
        print('error per dof: {0}'.format(terr))
    errs = (perr / abs(popt))[:3]
    if terr > 1.5:
        print('error per dof too large: {0}'.format(terr))
        print('relative errors too large: {0}'.format(errs))
        if return_error:
            return None, terr
        else:
            return None
    else:
        if return_error:
            return popt[:3], terr
        else:
            return popt[:3]

