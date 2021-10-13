import numpy as np
import pandas as pd

# one dimensional linear interpolation non uniform grid


def find_bbs(x, xs):
    ia, ib = 0, xs.shape[0] - 1
    if x <= xs[ia]:
        return ia, ia + 1
    if x >= xs[ib]:
        return ib - 1, ib
    while (ib - ia) > 1:
        im = (ib + ia) // 2
        if (xs[ib] - x) * (x - xs[im]) > 0:
            ia = im
        elif x == xs[ib]:
            return ib - 1, ib
        elif x == xs[im]:
            return im, im + 1
        else:
            ib = im
    return ia, ib


def get_function_values(x, xs, ys):
    ia, ib = find_bbs(x, xs)
    xa, xb = xs[ia], xs[ib]
    ya, yb = ys[ia], ys[ib]
    if xa != xb:
        y = ya + (yb - ya) * (x - xa) / (xb - xa)
    else:
        y = ya
    return y


def interpolate_nonuniform_linear(xgrid, xvalues, yvalues, non_unique_x=True):
    xvalues, yvalues = np.array(xvalues), np.array(yvalues)
    if non_unique_x:
        xvalues, yvalues = retrieve_unique_x_maxy(xvalues, yvalues)
    else:
        _, ix = np.unique(xvalues, return_index=True)
        xvalues, yvalues = xvalues[ix], yvalues[ix]
    ix = np.argsort(xvalues)
    xvalues, yvalues = xvalues[ix], yvalues[ix]
    ygrid = [get_function_values(x, xvalues, yvalues) for x in xgrid]
    ygrid = np.array(ygrid)
    return ygrid


def retrieve_unique_x_maxy(xvalues, yvalues):
    df_data = pd.DataFrame([xvalues, yvalues], index=["x", "y"]).T
    dfr = df_data.groupby("x").apply(lambda item: item["y"].max())
    return dfr.index, dfr.values


def integral_linear(xvalues, yvalues):
    xvalues, yvalues = np.array(xvalues), np.array(yvalues)
    ix = np.argsort(xvalues)
    xvalues, yvalues = xvalues[ix], yvalues[ix]

    s = 0
    for xa, xb, ya, yb in zip(xvalues[:-1], xvalues[1:], yvalues[:-1], yvalues[1:]):
        s += 0.5 * (ya + yb) * (xb - xa)
    return s
