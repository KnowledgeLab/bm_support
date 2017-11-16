import numpy as np
from numpy import min
import distance as d
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import unidecode
from pandas import DataFrame
from os.path import expanduser

# t1, t2 = tokenize_clean(s1, target_words), tokenize_clean(s2, target_words)


def tokenize_clean(s, target_words):
    s2 = unidecode.unidecode(s)
    s2 = s2.lower()
    words = s2.split(',')
    words2 = list(filter(lambda w: any([t in w for t in target_words]), words))
    tokenizer = RegexpTokenizer(r'\w+')
    phrases = [tokenizer.tokenize(w) for w in words2]
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
    # the distance will be at least 1.0
    dist_agg = [1.0]
    for u1 in t1:
        for u2 in t2:
            dist_agg.append(compute_phrase_distance(u1, u2, verbose))
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


def generate_fnames(j):
    return {'fname': expanduser('~/tmp/affiliation_alpha_beta_{0}.csv.gz'.format(j))}


def wrapper_disabmi(dfa, dfb, targets, fname, full_report=False):
    r = disambiguator_vec(dfa, dfb, targets, full_report)
    r.to_csv(fname, compression='gzip')