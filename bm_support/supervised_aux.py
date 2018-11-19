from numpy.random import RandomState
from sklearn.model_selection import train_test_split
from .supervised import simple_stratify
from .supervised import select_features_dict, logit_pvalue, report_metrics
import numpy as np

metric_selector = dict(zip(['corr', 'accuracy', 'precision', 'recall', 'f1'], range(5)))


def study_sample(seed, dfw, dist, feature_dict,
                 metric_mode, model_type, n_subtrials, n_estimators,
                 log_reg_dict={'min_log_alpha': -2., 'max_log_alpha': 2.},
                 verbose=False):
    nmax = 10000
    min_log_alpha, max_log_alpha = log_reg_dict['min_log_alpha'], log_reg_dict['max_log_alpha']
    feature_dict_inv = {}
    for k, v in feature_dict.items():
        feature_dict_inv.update({x: k for x in v})

    rns = RandomState(seed)
    df_train, df_testgen = train_test_split(dfw, test_size=0.4,
                                            random_state=rns, stratify=dfw[dist])

    df_valid, df_test = train_test_split(df_testgen, test_size=0.5,
                                         random_state=rns)

    vc = df_train[dist].value_counts()
    if verbose and vc.shape[0] < 5:
        print('*** df_train dist vc')
        print(vc)
    # training on the normalized frequencies
    df_train2 = simple_stratify(df_train, dist, seed, ratios=(2, 1, 1))

    if model_type == 'rf':
        param_dict = {'n_estimators': n_estimators, 'max_features': None, 'n_jobs': 1}
    else:
        param_dict = {'n_jobs': 1}

    meta_agg = []
    if model_type == 'rf':
        enums = rns.randint(nmax, size=n_subtrials)
    elif model_type == 'lr':
        delta = (max_log_alpha - min_log_alpha) / n_subtrials
        enums = 1e1 ** np.arange(min_log_alpha, max_log_alpha, delta)
    else:
        enums = []

    for ii in enums:
        if model_type == 'rf':
            param_dict['random_state'] = ii
        elif model_type == 'lr':
            param_dict['C'] = ii

        # for random forest different seed yield different models, for logreg models are penalty-dependent
        cfeatures, cmetrics, cvector_metrics, model_ = select_features_dict(df_train2, df_test, dist,
                                                                            feature_dict,
                                                                            model_type=model_type,
                                                                            max_features_consider=8,
                                                                            metric_mode=metric_mode,
                                                                            model_dict=param_dict,
                                                                            verbose=verbose)
        vscalar, vvector = report_metrics(model_, df_valid[cfeatures], df_valid[dist])
        ii_dict = {}
        ii_dict['run_par'] = seed
        ii_dict['current_features'] = cfeatures
        ii_dict['current_metrics'] = cmetrics
        ii_dict['current_vector_metrics'] = cvector_metrics
        ii_dict['validation_scalar_metrics'] = vscalar
        ii_dict['validation_vector_metrics'] = vvector
        ii_dict['model'] = model_
        ii_dict['corr_all'] = dfw[cfeatures + [dist]].corr()[dist]
        ii_dict['corr_all_test'] = df_test[cfeatures + [dist]].corr()[dist]
        ii_dict['corr_all_valid'] = df_valid[cfeatures + [dist]].corr()[dist]

        if model_type == 'lr':
            ii_dict['pval_errors'] = logit_pvalue(model_, df_train2[cfeatures])

        meta_agg.append(ii_dict)

    vscalar_mm = [x['validation_scalar_metrics'][metric_selector[metric_mode]] for x in meta_agg]
    vscalar = [x['validation_scalar_metrics'] for x in meta_agg]
    vvector = [x['validation_vector_metrics'] for x in meta_agg]
    index_best_run = np.argmax(vscalar_mm)
    best_features = meta_agg[index_best_run]['current_features']
    best_feature_groups = [feature_dict_inv[f] for f in best_features]

    report_dict = {}
    report_dict['best_features'] = best_features
    report_dict['best_feature_groups'] = best_feature_groups
    report_dict['max_scalar_mm'] = np.max(vscalar_mm)
    report_dict['vscalar'] = vscalar
    report_dict['vvector'] = vvector
    report_dict['corr_all'] = dfw[best_features + [dist]].corr()[dist]
    report_dict['corr_all_test'] = df_test[best_features + [dist]].corr()[dist]
    report_dict['corr_all_valid'] = df_valid[best_features + [dist]].corr()[dist]
    if model_type == 'lr':
        report_dict['pval_errors'] = meta_agg[index_best_run]['pval_errors']

    return report_dict, meta_agg
