from sklearn.cluster import KMeans
from numpy import arange, array, argmax, mean, sum, min, max, log, std, sqrt
from numpy.random import RandomState


def calc_wk(sample, clf=None, nc=2):
    if not clf:
        km = KMeans(nc)
        args = km.fit_predict(sample)
    clusters = [sample[args == k] for k in range(nc)]
    mus = [mean(c, axis=0) for c in clusters]
    dks = [sum((c - mu)**2) for c, mu in zip(clusters, mus)]
    wk = sum(dks)
    return wk


def gap_stat(data, nc, nsamples=10, seed=17):
    """
    # calculate gap statistic for data for nc clusters

    Tibshirani, R., Walther, G. and Hastie, T. (2001),
    Estimating the number of clusters in a data set via the gap statistic.
    Journal of the Royal Statistical Society: Series B (Statistical Methodology), 63: 411–423.
    doi:10.1111/1467-9868.00293

    :param data:
    :param nc:
    :param nsamples:
    :param seed:
    :return:
    """

    wk_r = calc_wk(data, nc=nc)
    # protection from zeros
    if wk_r == 0:
        wk_r = 1e-10
    # generate uniform samples
    nsize = data.shape[0]*data.shape[1]
    mins = min(data, axis=0)
    maxs = max(data, axis=0)
    rns = RandomState(seed)
    random_samples = [rns.uniform(size=nsize).reshape(data.shape)*(maxs-mins) + mins for k in range(nsamples)]
    wks = list(map(lambda x: calc_wk(x, nc=nc), random_samples))
    # protection from zeros
    wks = array([wk if wk > 0 else 1e-20 for wk in wks])
    log_wks = log(wks)
    sk = std(log_wks)*sqrt(1. + 1./nsamples)
    gap = mean(log_wks) - log(wk_r)
    return gap, sk


def choose_nc(data, nc_max=3, verbose=False):
    if nc_max > 1:
        if len(data) > nc_max:
            gstats = [gap_stat(data, k) for k in arange(1, nc_max+2)]
            ds = [x[0] - y[0] + y[1] for x, y in zip(gstats[:-1], gstats[1:])]
            if any(array(ds) > 0):
                nc_opt = argmax(array(ds) > 5e-2) + 1
            else:
                nc_opt = -1
                if verbose:
                    print(ds)
                    print('consider re-running choose_nc() with increased nc_max')
        else:
            nc_opt = 1
    else:
        raise ValueError('nc_max should be greater than 1, value {0} supplied instead'.format(nc_max))
    return nc_opt
