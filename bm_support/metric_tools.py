import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def compute_topk_thr(y_test, y_pred, test_thr=25, rec_level=0.5):
    correct_thr = np.percentile(y_test, test_thr)

    y_test_bin = (y_test < correct_thr).astype(int)
    base_prec = np.mean(y_test_bin)
    top_ps = np.arange(0.01, 1.0, 0.01)
    precs = []
    recs = []
    f1s = []
    for p_level in top_ps:
        p_thr = np.percentile(y_pred, 100 * (1 - p_level))
        y_pred_bin = (y_pred < p_thr).astype(int)
        precs.append(precision_score(y_test_bin, y_pred_bin))
        recs.append(recall_score(y_test_bin, y_pred_bin))
        f1s.append(f1_score(y_test_bin, y_pred_bin))
    precs = np.array(precs)
    recs = np.array(recs)
    # greater - better
    delta_precs = (precs - base_prec) / base_prec
    delta_recs = recs
    ii = np.argmin(delta_recs > rec_level)
    best_prec, best_rec = precs[ii], recs[ii]
    return best_prec, best_rec, base_prec, precs, recs
