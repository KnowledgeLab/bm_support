import numpy as np
from numpy import min
import distance as d
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import unidecode
from pandas import DataFrame
from sklearn.cluster import KMeans
import  Levenshtein as lev
from os.path import expanduser

# t1, t2 = tokenize_clean(s1, target_words), tokenize_clean(s2, target_words)


def tokenize_clean(s, target_words=None):
    s2 = unidecode.unidecode(s)
    s2 = s2.lower().strip('.')
    words = s2.split(',')
    # if target_words:
    #     words = list(filter(lambda w: any([t in w for t in target_words]), words))
    tokenizer = RegexpTokenizer(r'\w+')
    phrases = [tokenizer.tokenize(w) for w in words]
    phrases2 = [f7([w for w in s if w not in stopwords.words('english')]) for s in phrases]
    return phrases2


def f7(seq):
    """
    https://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-in-whilst-preserving-order
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def prod_metric_matrix(t1, t2):
    if not t1 or not t2:
        return None
    m = np.array([[d.nlevenshtein(w1, w2, method=2) for w1 in t1] for w2 in t2])
    if m.shape[0] > m.shape[1]:
        m = m.T
    return m


def norm(n):
    # maximum \sum ( (i - \sigma_i)^2)
    # NB : n should be > 1
    k = n // 2
    if n % 2:
        # n odd
        f = 4 * k * (k + 1) * (2 * k + 1) // 3
    else:
        # n even
        f = 2 * k * (2 * k + 1) * (2 * k - 1) // 3
    return f


def norm2(n):
    # assuming uqique pairs a a'
    # for each a' exists unique closest a
    k = n - 1
    f = k * (k + 1) * (2 * k + 1) // 6
    return f


def permutation_metric(m, verbose=False):
    # m is of shape ndim x ndim' where ndim <= ndim'
    ndim = m.shape[0]
    if m is not None and ndim > 0:
        args = np.argmin(m, axis=1)
        dist_map = np.sum(m[np.arange(ndim), args] ** 2) / ndim
        args2 = np.argmin(m[:, sorted(args)], axis=1)
        if ndim > 1:
            dist_perm = np.sum((np.arange(ndim) - args2) ** 2) / norm(ndim)
        else:
            dist_perm = 0.
        rho = (0.5 * (dist_map + dist_perm)) ** 0.5
    else:
        return 1.0
    return rho


def compute_phrase_distance(a1, a2, synonyms=[], verbose=False):
    # a1 and a2 are two lists of strings
    m = prod_metric_matrix(a1, a2)
    return permutation_metric(m, verbose)


def clean_compute_metric(s1, s2, target_words, verbose=False):
    t1, t2 = tokenize_clean(s1, target_words), tokenize_clean(s2, target_words)
    # print(t1, t2)
    # the distance will be at least 1.0
    dist_agg = [1.0]
    for u1 in t1:
        for u2 in t2:
            d = compute_phrase_distance(u1, u2, verbose)
            dist_agg.append(d)
            # print(u1, u2, d)
    return min(dist_agg)


def disambiguator(item, entity_vector, targets):
    metric = []
    for a in entity_vector:
        rho = clean_compute_metric(item, a, targets)
        metric.append(rho)
        if rho == 0:
            break
    k = np.argmin(metric)
    rho = metric[k]
    return rho, k


def disambiguator_vec(dfa, dfb, targets, full_report=False):
    """

    :param dfa: DataFrame with two columns: c1 and c2
                dfa[c1] contains the id_a (positive int)
                dfa[c2] contains the string (str)
    :param dfb: DataFrame with two columns: c1 and c2
                dfb[c1] contains the id_b (positive int)
                dfb[c2] contains the string (str)
    :param targets:
    :return:
    NB: it is assumed that ids_b are positive integers
    """

    agg = []
    ids_b, strings_b = dfb.values[:, 0], dfb.values[:, 1]

    for id_a, str_a in dfa.values:
        score, best_j = disambiguator(str_a, strings_b, targets)
        if score < 0.05:
            id_b = ids_b[best_j]
            str_b = strings_b[best_j]
        else:
            id_b = -1
            str_b = None
        if full_report:
            item = id_a, id_b, str_a, str_b, score
        else:
            item = id_a, id_b
        agg.append(item)
    if full_report:
        dfr = DataFrame(np.array(agg),
                        columns=[dfa.columns[0], dfb.columns[0], dfa.columns[1], dfb.columns[1], 'score'])
    else:
        dfr = DataFrame(np.array(agg),
                        columns=[dfa.columns[0], dfb.columns[0]])
    return dfr


def split_df(df, n_pieces):
    inds = [df.index[k::n_pieces] for k in range(n_pieces)]
    dfs = [df.loc[ind] for ind in inds]
    return dfs


def generate_fnames(tmp_path, j):
    return {'fname': '{0}/affiliation_alpha_beta_{1}.csv.gz'.format(tmp_path, j)}


def wrapper_disabmi(dfa, dfb, targets, fname, full_report=False):
    r = disambiguator_vec(dfa, dfb, targets, full_report)
    r.to_csv(fname, compression='gzip')


def cluster_strings(ids, strings, targets, decision_thr=0.1, max_it=None, n_classes=2, tol=1e-6, seed=0,
                    verbose=False):
    id_cluster_dict = {}
    k = 0
    while ids.size > 1:
        cur_id, ids = ids[0], ids[1:]
        cur_str, strings = strings[0], strings[1:]
        dists = np.array([clean_compute_metric(cur_str, a, targets) for a in strings])
        n_c = n_classes if ids.size > 2 else 1
        km = KMeans(n_c, tol=tol, random_state=seed)
        args = km.fit_predict(dists.reshape(-1, 1))
        unis = np.unique(args)
        centers = np.concatenate([km.cluster_centers_[k] for k in unis])
        proximal_class = np.argmin(centers)
        # if verbose:
        #     print(sorted(centers), len(strings), sum(args == proximal_class), sum(args != proximal_class))
        if centers[proximal_class] < decision_thr:
            if verbose:
                print('{0} strings clustered'.format(sum(args == proximal_class) + 1))
            proximal_indices = np.where(args == proximal_class)
            distant_indices = np.where(args != proximal_class)
            id_cluster_dict[k] = [cur_id] + list(ids[proximal_indices])
            ids = ids[distant_indices]
            strings = strings[distant_indices]
        else:
            id_cluster_dict[k] = [cur_id]
        k += 1
        if max_it and k > max_it:
            break

        # if verbose:
        #     print(sum([len(x) for x in id_cluster_dict.values()]) + len(strings))
    id_cluster_dict[k] = [cur_id]
    return id_cluster_dict


def dict_to_array(ddict):
    # should work with list and numpy arrays (1D and 2D)
    # only checked for lists
    #TODO debug for arrays
    """
        ddict contains arrays of size n \times k_i
        final array has size (n+1) \times \sum k_i
    """
    keys = list(ddict.keys())
    if isinstance(ddict[keys[0]], list):
        arrays_list = [np.array(ddict[k]) for k in keys]
    else:
        arrays_list = [ddict[k] for k in keys]
    if  isinstance(ddict[keys[0]], np.ndarray) and len(ddict[keys[0]].shape) > 1:
        arr = np.concatenate(arrays_list, axis=1)
        keys_list = [[k]*ddict[k].shape[1] for k in keys]
    else:
        arr = np.concatenate(arrays_list).reshape(-1, 1)
        keys_list = [[k]*len(ddict[k]) for k in keys]
    keys_arr = np.concatenate(keys_list).reshape(-1, 1)
    print(arr.shape, keys_arr.shape)
    final_array = np.concatenate([keys_arr, arr], axis=1)
    return final_array


def split_string(s):
    s2 = unidecode.unidecode(s).lower()
    phrases = s2.split(',')
    return phrases


def tokenize_phrase(s):
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(s)
    clean_words = [w for w in words if w not in stopwords.words('english')]
    return clean_words


def nlevenstein_root(s, s2, foo):
    """
    normed levenshtein metric
    :param s:
    :param s2:
    :param foo:
    :return:
    """
    if len(s) > len(s2):
        s, s2 = s2, s
    diff = len(s2) - len(s) + 1
    m = min([foo(s, s2[k:(k+len(s))]) for k in range(diff)])
    return m


def ndistance(s1, s2):
    return lev.distance(s1, s2)/min([len(s1), len(s2)])


def metric_to_words(m, distance_func, decision_thr=0.1, verbose=False):
    dim1, dim2 = m.shape
    qw1 = ''.join([chr(j+ord('a')) for j in range(dim1)])
    complement_alphabet = [chr(j + dim1 + ord('a')) for j in range(dim2)]
    qw2 = []
    mms = []

    for r, j in zip(m.T, np.argmin(m, axis=0)):
        if r[j] < decision_thr:
            qw2.append(qw1[j])
        else:
            qw2.append(complement_alphabet.pop())
        mms.append(r[j])
    qw2 = ''.join(qw2)

    words_discrepancy_norm = np.sum(np.min(m, axis=0)**2/m.shape[1])**0.5
    # e.g. ndistance(qw1, qw2, method=2)
    phrase_norm = distance_func(qw1, qw2)

    return phrase_norm, words_discrepancy_norm


def meta_distance(mw1, mw2, foo, sigma=0.5):
    if isinstance(mw1, str) and isinstance(mw2, str):
        r = foo(mw1, mw2)
    else:
        m = np.array([[meta_distance(w1, w2, foo) for w1 in mw1] for w2 in mw2])
        da, db = metric_to_words(m, foo)
        r = ((da**2 + db**2)/2.)**0.5
    return r
