from pandas import Series, DataFrame, merge
from numpy import std, mean, corrcoef
from os.path import expanduser
from matplotlib.pyplot import subplots

up = 'up'
dn = 'dn'
o_columns = [up, dn]


def compute_correlations(args_lincs_std_list, args_reports_list, cutting_schedule, extra_masks):
    results = []
    cutting_column, levels = cutting_schedule

    level_mean, level_lo, level_hi = cutting_column + '_mean', cutting_column + '_lo', cutting_column + '_hi'

    for args, df_exp in args_lincs_std_list:
        m0 = Series([True]*df_exp.shape[0], df_exp.index)
        for c, thr, foo in extra_masks['exp']:
            m = (foo(df_exp[c], thr))
            m0 &= m
        dfl_mean_std2 = df_exp.loc[m0].copy()
        print('total number of degenerate (diff cell lines) claims {0}; {1}'.format(m.shape, sum(m)))
        # calculate mean and std across cell lines
        dfl_mean_std3 = dfl_mean_std2.groupby([up, dn]).apply(lambda x: Series([mean(x['mean']), std(x['mean'])],
                                                                               index=['imean', 'istd'])).reset_index()
        # choose reports
        reps = list(filter(lambda x: args.items() <= x[0].items(), args_reports_list))
        for args_reps, df_lit in reps:
            print(args_reps)
            m0 = Series([True]*df_lit.shape[0], df_lit.index)
            for c, thr, foo in extra_masks['literature']:
                m = (foo(df_lit[c], thr))
                m0 &= m
            df_aux = df_lit.loc[m0].copy()
            print('given lit df', df_lit.shape, df_aux.shape)
            sm_last = -1
            for lo, hi in zip(levels[:-1], levels[1:]):
                mlo = (df_aux[cutting_column] >= lo)
                mhi = (df_aux[cutting_column] < hi)
                sm = sum(mlo & mhi)
                print('lo and hi:', lo, hi, sm)
                if sm > 2:
                    df2 = df_aux.loc[mlo & mhi].copy()

                    df_cmp = merge(df2[[up, dn, 'freq', 'pi_last']], dfl_mean_std3[[up, dn, 'imean']],
                                   how='inner', on=o_columns)
                    print(df_lit.shape, df_cmp.shape, dfl_mean_std3.shape)

                    print('level: {3:.2f}; experimentally accepted pairs {0} out of {1}; '
                          'after merging onto claims : {2}'.format(sum(mlo & mhi), mlo.shape[0], df_cmp.shape[0], lo))

                    x = df_cmp['imean'].values
                    y = df_cmp['freq'].values
                    z = df_cmp['pi_last'].values
                    size = df_cmp.shape[0]

                    cov_freq_ = corrcoef(x, y)[0, 1]
                    cov_flat_ = corrcoef(x, z)[0, 1]

                    exp_mean = x.mean()
                    exp_std = x.std()
                    lit_mean = y.mean()
                    lit_std = y.std()

                    if sm != sm_last:
                        res_dict = {}
                        res_dict.update(args_reps)
                        res_dict.update(dict(zip(['size', level_mean, level_lo, level_hi,
                                                  'cor_freq', 'cor_model',
                                                  'e_mean', 'e_std', 'l_mean', 'l_std'],
                                                 [size, 0.5 * (lo + hi), lo, hi,
                                                  cov_freq_, cov_flat_,
                                                  exp_mean, exp_std, lit_mean, lit_std])))
                        results.append(res_dict)
                    sm_last = sm

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


def plot_subsets(df, masks, ycols, markers, marker_names, title, location='upper left', save_file=False):
    lwidth = 1.4
    msize = 16
    xcol = 'delta_year'
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
