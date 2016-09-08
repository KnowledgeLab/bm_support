from numpy import array, arange, histogram, argmax, greater, nan
from scipy.signal import argrelextrema


def trim_data(data, n_bins=10):
    # find left and right bounds such that
    # the number of counts is above average (deviates from uniform)

    n = data.shape[0]
    cnts, xcoords = histogram(data, bins=n_bins)
    mean_counts = float(n) / n_bins
    lb, rb = argmax(cnts > mean_counts), argmax(cnts[::-1] > mean_counts)
    return xcoords[lb], xcoords[-(rb+1)]


def histogram_range(data, xr, n_bins=20):
    # bin data in n_bins within xr
    delta = (xr[1] - xr[0]) / n_bins
    bins = arange(xr[0], xr[1] + delta, delta)
    cnts, xcoords = histogram(data, bins)
    return cnts, xcoords


def find_max_neighbourhood(data, i):
    # return the size of the neighbourhood
    # for which data[i] is a local maximum
    # and the sum of data over the neighbourhood
    j = 0
    k = 0
    while i + k + 1 < data.shape[0] and data[i + k + 1] < data[i + k]:
        k += 1
    while i - j - 1 > 0 and data[i - j - 1] < data[i - j]:
        j += 1
    if k * j == 0:
        k = 0
        j = 0
    return 0.5 * (k + j), sum(data[i - j:i + k + 1])


def analyse_local_maxima(data, xr, n_bins=20,
                         n_ext=2, alpha=0.2, beta=0.5,
                         gamma=0.2):
    cnts, xcoords = histogram_range(data, xr, n_bins)
    # the maxima are either a) in the interior or b) the boundaries
    # a) interior of cnts
    # get the extrema indices
    inds_ext = argrelextrema(cnts, greater)[0]
    # print type(inds_ext), inds_ext
    if len(inds_ext) > 0:
        # get the extrema values
        cnts_ext = cnts[inds_ext]
        # rank the extrema
        ranks_ext = cnts_ext.argsort()

        # take top n_ext extrema
        # provide top n_ext indices in reverse (descending) order
        # arg_sort is in ascending order
        top_inds = inds_ext[ranks_ext[:-n_ext - 1:-1]]
        fs = cnts[top_inds]
        nhoods = array([find_max_neighbourhood(cnts, pos) for pos in top_inds]).T
        report = True, 'Posterior acceptable'
        ans = xcoords[top_inds[0]]
        if top_inds.shape[0] > 1 and alpha * fs[0] < fs[1]:
            if beta * nhoods[1, 0] < nhoods[1, 1]:
                if abs((xcoords[top_inds[0]] - xcoords[top_inds[1]])/(xr[1] - xr[0])) > gamma:
                    report = False, 'Posterior unacceptable: at least two prominnent peaks'
                    ans = nan
    else:
        report = False, 'Posterior unacceptable: extremum at the boundary'
        ans = xcoords[argmax(cnts)]
    return ans, report
