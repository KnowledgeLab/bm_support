from pandas import Series, DataFrame, merge, concat
from numpy import std, mean, corrcoef, array, tile, arctanh, tanh, concatenate, log10, arange
from scipy.stats import t
from datahelpers.dftools import get_multiplet_to_int_index
from os.path import expanduser
from matplotlib.pyplot import subplots
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

up = 'up'
dn = 'dn'
o_columns = [up, dn]


def corr_errs(c, n, confidence=0.95):
    if n > 3:
        rho = arctanh(c)
        se = 1./(n-3)**0.5
        h = se * t.ppf(0.5*(1 + confidence), n-1)
        left_, right_ = rho - h, rho + h
        left, right = tanh(left_), tanh(right_)
        return c - left, right - c
    else:
        return None


def round_to_delta(x, delta, x0=0.0):
    k = int((x - x0)/delta + 0.5)
    return x0+k*delta


def obtain_crosssection(args_lincs_std_list, args_reports_list, masks):
    results = []
    for args, df_exp in args_lincs_std_list:
        m0 = Series([True]*df_exp.shape[0], df_exp.index)
        for c, thr, foo in masks['exp']:
            m = (foo(df_exp[c], thr))
            m0 &= m
        dfl_mean_std2 = df_exp.loc[m0].copy()
        dfl_mean_std3 = dfl_mean_std2.groupby([up, dn]).apply(lambda x: Series([mean(x['mean']), std(x['mean'])],
                                                                               index=['imean', 'istd'])).reset_index()
        if 'exp_' in masks.keys():
            for c, thr, foo in masks['exp_']:
                m = (foo(dfl_mean_std3[c], thr))
                m0 &= m
            dfl_mean_std3 = dfl_mean_std3.loc[m0].copy()

        reps = list(filter(lambda x: args.items() <= x[0].items(), args_reports_list))
        for args_reps, df_lit in reps:
            m0 = Series([True]*df_lit.shape[0], df_lit.index)
            for c, thr, foo in masks['literature']:
                m = (foo(df_lit[c], thr))
                m0 &= m
            df_aux = df_lit.loc[m0].copy().reset_index()
            df_cmp = merge(df_aux, dfl_mean_std3[[up, dn, 'imean']],
                           how='inner', on=o_columns)
            print('literature datapoints {0} out of {1}; '
                  'after merging onto claims : {2}'.format(df_aux.shape[0],
                                                           df_lit.shape[0], df_cmp.shape[0]))
            results.append((args_reps, df_cmp))
    return results


def obtain_crosssection2(args_lincs_std_list, args_reports_list, extra_masks, verbose=False):
    df_acc = DataFrame()
    for args, df_exp in args_lincs_std_list:
        m0 = Series([True]*df_exp.shape[0], df_exp.index)
        for c, thr, foo in extra_masks['exp']:
            m = (foo(df_exp[c], thr))
            m0 &= m
        dfl_mean_std2 = df_exp.loc[m0].copy()
        if verbose:
            print('total number of statement passing experimental masking '
                  '(before cell-line group-by) {0} out of {1}'.format(sum(m0), m0.shape[0]))
        # calculate mean and std across cell lines
        dfl_mean_std3 = dfl_mean_std2.groupby([up, dn]).apply(lambda x: Series([mean(x['mean']), std(x['mean'])],
                                                                               index=['imean', 'istd'])).reset_index()
        if 'exp_' in extra_masks.keys():
            m0 = Series([True] * dfl_mean_std3.shape[0], dfl_mean_std3.index)

            for c, thr, foo in extra_masks['exp_']:
                print(c, thr)
                m = (foo(dfl_mean_std3[c], thr))
                m0 &= m
            dfl_mean_std3 = dfl_mean_std3.loc[m0].copy()

        if verbose:
            print('total number of statement passing experimental masking '
                  'after cell-line group-by {0} out of {1}'.format(sum(m0), m0.shape[0]))
        # choose reports
        reps = list(filter(lambda x: args.items() <= x[0].items(), args_reports_list))
        for args_reps, df_lit in reps:
            m0 = Series([True]*df_lit.shape[0], df_lit.index)
            for c, thr, foo in extra_masks['literature']:
                m = (foo(df_lit[c], thr))
                m0 &= m
            # [up, dn, 'freq', 'pi_last']
            tmp = df_lit.loc[m0].copy()
            tmp2 = DataFrame(Series(args_reps)).T
            tmp3 = DataFrame(tile(tmp2.values, (len(tmp.index), 1)),
                             index=tmp.index, columns=tmp2.columns)
            tmp4 = merge(tmp, tmp3, left_index=True, right_index=True)
            df_cmp = merge(tmp4, dfl_mean_std3[[up, dn, 'imean']], how='inner', on=o_columns)
            if verbose:
                print('literature datapoints {0} '
                      'after merging onto claims : {1}'.format(df_lit.shape[0], df_cmp.shape[0]))
            if verbose:
                print('df_cmp {0} '
                      'df_acc : {1}'.format(df_cmp.shape, df_acc.shape))
            df_acc = concat([df_cmp, df_acc])
    if verbose:
        print('final df_acc : {0}'.format(df_acc.shape))
    return df_acc


#TODO extend cutting_schedule to []

def compute_correlations(args_lincs_std_list, args_reports_list, cutting_schedule, extra_masks,
                         round_delta_y=False, delta=0.1, x0=0.0, verbose=False):
    results = []
    cutting_column, levels = cutting_schedule

    level_mean, level_lo, level_hi = cutting_column + '_mean', cutting_column + '_lo', cutting_column + '_hi'

    for args, df_exp in args_lincs_std_list:
        m0 = Series([True]*df_exp.shape[0], df_exp.index)
        for c, thr, foo in extra_masks['exp']:
            m = (foo(df_exp[c], thr))
            m0 &= m
        dfl_mean_std2 = df_exp.loc[m0].copy()
        if verbose:
            print('total number of statement passing experimental masking '
                  '(before cell-line group-by) {0} out of {1}'.format(sum(m0), m0.shape[0]))
        # calculate mean and std across cell lines
        dfl_mean_std3 = dfl_mean_std2.groupby([up, dn]).apply(lambda x: Series([mean(x['mean']), std(x['mean'])],
                                                                               index=['imean', 'istd'])).reset_index()
        if 'exp_' in extra_masks.keys():
            m0 = Series([True] * dfl_mean_std3.shape[0], dfl_mean_std3.index)

            for c, thr, foo in extra_masks['exp_']:
                m = (foo(dfl_mean_std3[c], thr))
                m0 &= m
            dfl_mean_std3 = dfl_mean_std3.loc[m0].copy()

        if verbose:
            print('total number of statement passing experimental masking '
                  'after cell-line group-by {0} out of {1}'.format(sum(m0), m0.shape[0]))
        # choose reports
        reps = list(filter(lambda x: args.items() <= x[0].items(), args_reports_list))
        for args_reps, df_lit in reps:
            m0 = Series([True]*df_lit.shape[0], df_lit.index)
            for c, thr, foo in extra_masks['literature']:
                m = (foo(df_lit[c], thr))
                m0 &= m
            df_aux = df_lit.loc[m0].copy()
            for lo, hi in zip(levels[:-1], levels[1:]):
                mlo = (df_aux[cutting_column] >= lo)
                mhi = (df_aux[cutting_column] < hi)
                sm = sum(mlo & mhi)
                df_cmp = merge(df_aux.loc[mlo & mhi, [up, dn, 'freq', 'pi_last', cutting_column]],
                               dfl_mean_std3[[up, dn, 'imean']],
                               how='inner', on=o_columns)
                if verbose:
                    print('levels: [{3:.2f}; {4:.2f}); literature datapoints {0} out of {1}; '
                          'after merging onto claims : {2}'.format(sm, df_lit.shape[0], df_cmp.shape[0], lo, hi))
                size = df_cmp.shape[0]
                if size > 4:
                    x = df_cmp['imean'].values
                    y = df_cmp['freq'].values
                    z = df_cmp['pi_last'].values
                    if round_delta_y and delta > 0.:
                        y = array(list(map(lambda w: round_to_delta(w, delta, x0), y)))
                        z = array(list(map(lambda w: round_to_delta(w, delta, x0), z)))

                    std_x = std(x)
                    std_y = std(y)
                    std_z = std(z)
                    if std_x > 0 and std_y > 0 and std_z > 0:
                        cov_freq_ = corrcoef(x, y)[0, 1]
                        cov_flat_ = corrcoef(x, z)[0, 1]

                        cov_freq_err = (1. - cov_freq_**2)/(size**0.5)
                        cov_flat_err = (1. - cov_flat_**2)/(size**0.5)

                        cov_freq_err_left, cov_freq_err_right = corr_errs(cov_freq_, size)
                        cov_flat_err_left, cov_flat_err_right = corr_errs(cov_flat_, size)

                        exp_mean = x.mean()
                        exp_std = x.std()
                        lit_mean = y.mean()
                        lit_std = y.std()

                        res_dict = {}
                        res_dict.update(args_reps)
                        res_dict.update(dict(zip(['size', level_mean, level_lo, level_hi,
                                                  'cor_freq', 'cor_model',
                                                  'cor_freq_err_left', 'cor_freq_err_right',

                                                  'cor_model_err_left', 'cor_model_err_right',
                                                  'e_mean', 'e_std', 'l_mean', 'l_std'],
                                                 [size, 0.5 * (lo + hi), lo, hi,
                                                  cov_freq_, cov_flat_,
                                                  cov_freq_err_left, cov_freq_err_right,
                                                  cov_flat_err_left, cov_flat_err_right,
                                                  exp_mean, exp_std, lit_mean, lit_std])))
                        results.append(res_dict)

    df_out = DataFrame.from_dict(results)
    return df_out


def extract_xy_werrors(df, subset_spec, xcolumn, ycolumn, x_ext_names=None, y_ext_names=None):
    """

    :param df:
    :param subset_spec:
    :param xcolumn:
    :param ycolumn:
    :param x_ext_names: ['lo', 'mean', 'hi']
    :param y_ext_names:
    :return:
    """
    if x_ext_names:
        xcol_names = ['{0}_{1}'.format(xcolumn, ext) for ext in x_ext_names]
    else:
        xcol_names = [xcolumn]

    if y_ext_names:
        ycol_names = ['{0}_{1}'.format(xcolumn, ext) for ext in y_ext_names]
    else:
        ycol_names = [ycolumn]

    mask_acc = Series([True] * df.shape[0], index=df.index)
    for k, v in subset_spec.items():
        m = (df[k] == v)
        mask_acc = (mask_acc & m)

    dfr = df.loc[mask_acc].drop_duplicates(xcol_names + ycol_names)
    xs = [dfr[xc].values for xc in xcol_names]
    ys = [dfr[yc].values for yc in ycol_names]
    return xs, ys


def plot_subsets(df, masks, xcol, ycols, markers, marker_names, title, location='upper left',
                 logx=False, logy=False,
                 save_file=False):
    lwidth = 1.4
    msize = 16
    extensions = ['lo', 'mean', 'hi']

    arrays = [extract_xy_werrors(df, mask, xcol, ycol, extensions) for mask, ycol in zip(masks, ycols)]
    xs = [t[0][1] for t in arrays]
    ys = [t[1][0] for t in arrays]
    xerrs = [(x - xmi, xpl-x) for xmi, x, xpl in map(lambda z: z[0], arrays)]

    fig, ax = subplots(figsize=(5, 5))
    kwargs = {'alpha': 1, 'markersize': 0.5*msize, 'lw': lwidth, 'ls': 'None'}
    kwargs_list = [{**{'x': x, 'y': y, 'marker': m, 'xerr': [xe[0], xe[1]]}, **kwargs} for x, y, xe, m in
                   zip(xs, ys, xerrs, markers)]

    lines = [ax.errorbar(**pars) for pars in kwargs_list]
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')

    ax.grid(b=True, which='major', color='k', linestyle='--')
    ax.legend(lines, marker_names, loc=location, frameon=True,
              framealpha=1.0, facecolor='w', edgecolor='k', shadow=False, prop={'size': 12})
    ax.set_ylabel(r'$\rho\'s$', fontsize='x-large')
    ax.set_xlabel(xcol, fontsize='x-large')
    ax.set_title(title)
    if save_file:
        fig_title = title.replace(' ', '_')
        ffig_title = expanduser('~/data/kl/figs/') + fig_title + '.pdf'
        fig.savefig(ffig_title)


def pivot_cased_df(df, index_columns=o_columns, new_index='ii', columns_to_pivot='case', values_to_pivot='pi_last'):
    dfind = get_multiplet_to_int_index(df, index_columns, new_index)
    pivoted = dfind.pivot(index=new_index, columns=columns_to_pivot, values=values_to_pivot)
    pivoted = pivoted.rename(columns=dict(zip(pivoted.columns, ['est_{0}'.format(c) for c in pivoted.columns])))
    df0 = merge(pivoted, dfind, right_on=new_index, left_index=True, how='right').drop_duplicates(new_index)
    return df0


def linear_regression_routine(args_lincs_std_list, args_reports_list, feature_cols, extra_masks, fit_intercept=False):
    # strength of the signal window

    dfr = obtain_crosssection2(args_lincs_std_list, args_reports_list, extra_masks, verbose=True)
    dfp = pivot_cased_df(dfr)
    lr = linear_model.LinearRegression(fit_intercept=fit_intercept, normalize=False)
    # cross_val_predict returns an array of the same size as `y` where each entry
    # is a prediction obtained by cross validation:

    # skf = StratifiedKFold(n_splits=5)
    skf = KFold(n_splits=3)

    X = dfp[feature_cols].values
    y = dfp['imean'].values
    sc = MinMaxScaler()

    report = {'coefs': [], 'intercept': [], 'r2_score': [],
              'p_values': [], 't_stats': []}

    X = sc.fit_transform(X)

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        report['intercept'].append(lr.intercept_)
        report['coefs'].append(lr.coef_)
        report['r2_score'].append(r2_score(y_test, y_pred))
        t_stats = (sum((y_test - y_pred) ** 2) / (y_pred.shape[0] - 2))**0.5 / \
                  (sum((X_test - mean(X_test)) ** 2)) ** 0.5
        report['t_stats'].append(t_stats)
        p_values = 1. - stats.t.cdf(t_stats, df=y_pred.shape[0]-2)
        report['p_values'].append(p_values)
    return report
