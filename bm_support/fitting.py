from numpy import (
    array,
    sqrt,
    log,
    exp,
    pi,
    histogram,
    diag,
    abs,
    max,
    floor,
    ceil,
    min,
    arange,
    divide,
)
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from functools import partial

# this package find optimal lognormal fit from a frequentist point of view
# can be altered to fit any function


def lnormal(x, mu, sigma, a):
    """pdf of lognormal distribution"""

    return a * (
        exp(-((log(x) - mu) ** 2) / (2 * sigma ** 2)) / (x * sigma * sqrt(2 * pi))
    )


def lnormal_off(x, mu, sigma, a, t0):
    """pdf of lognormal distribution"""
    xmin = x.min()
    if x.min() > t0:
        x = (x - t0).copy()
    else:
        x = array(1e-10 * (t0 - xmin) * array([1] * x.shape[0]))
    return a * (
        exp(-((log(x) - mu) ** 2) / (2 * sigma ** 2)) / (x * sigma * sqrt(2 * pi))
    )


def fit_func(
    data,
    mode="pile",
    nbins=10,
    foo=lnormal,
    guess=(2.0, 1.0, 50),
    return_error=False,
    intelligent_guess=True,
    plot=False,
    error_max=3.0,
    verbose=False,
):
    """

    :param data:
    :param mode:
    :param nbins:
    :param foo:
    :param guess:
    :param return_error:
    :param intelligent_guess:
    :param plot:
    :param error_max:
    :param verbose:
    :return:
    """
    mean_points_per_bin = 20
    if mode == "pile":
        # nbins = min(nbins, )
        if intelligent_guess:
            min_data = floor(min(data))
            max_data = ceil(max(data))
            # have to to have at least k bins to fit for k parameters
            max_delta = int((max_data - min_data) / len(guess))
            delta = max(
                [int((max_data - min_data) * mean_points_per_bin / len(data)), 1.0]
            )
            delta = min([delta, max_delta])
            nbins = arange(min_data, max_data + 0.5 * delta, delta) - 0.5 * delta
            # nbins = nbins - 0.5*delta
            if verbose:
                print("min: {0} max: {1} delta: {2}".format(min_data, max_data, delta))
                print("bbs: {0}".format(nbins))

        y, bnds = histogram(data, bins=nbins, normed=False)
        if verbose:
            print("ys: {0}".format(y))

        delta = bnds[1] - bnds[0]
        x = bnds[1:] - bnds[0] - 0.5 * delta
        y = y / delta
    elif mode == "hist":
        x, y = data
        delta = x[1] - x[0]
    else:
        return None

    guess_ = list(guess)
    if intelligent_guess:
        guess_[2] = sum(y)
    if verbose:
        print("guess_: {0}".format(guess_))

    popt, pcov = curve_fit(foo, x, y, p0=guess_, method="lm")
    yf = foo(x, *popt)

    mask = yf > 1e-10
    if mask.all() and sum(mask) > 0:
        if verbose:
            print("delta: {1} {0}".format(yf, len(data)))
        if any(yf[mask] == 0):
            print("WTH: yf zeros {0}".format(yf[mask]))
        if sum(mask) == 0:
            print("WTH: sum_mask {0}".format(mask))
        tmp = divide(yf[mask] - y[mask], yf[mask], where=mask)
        terr = sum(tmp ** 2) ** 0.5
        terr = terr / sum(mask)
    else:
        return None

    if plot:
        fig, ax = plt.subplots()
        xfill = arange(x[0], x[-1], 0.05 * delta)
        ax.scatter(x, y)
        yf = foo(xfill, *popt)
        ax.plot(xfill, yf, linestyle="dashed", label="Fitted")

    if verbose:
        print("error per dof: {0}".format(terr))
    if terr > error_max:
        if verbose:
            print("error per dof too large: {0}".format(terr))
    if return_error:
        return list(popt) + [terr]
    else:
        return popt
