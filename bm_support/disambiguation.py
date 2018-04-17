import numpy as np
from numpy import min
import distance as d
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import unidecode
from pandas import DataFrame
import re
from numpy.random import RandomState
from functools import partial
import pathos.multiprocessing as mp
from copy import deepcopy


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


def clean_compute_metric2(s1, s2, target_words, verbose=False):
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


def split_string(s, sep=',', asciize=True, cut_brackets=True):
    if asciize:
        s = unidecode.unidecode(s).lower()
    if cut_brackets:
        s = re.sub('[\(\[].*?[\)\]]', '', s)
    s = re.sub('\s+', ' ', s).strip()
    phrases = [x.strip() for x in s.split(sep)]
    return phrases


def tokenize_phrase(s, boring_words=[]):
    words = s.strip().split(' ')
    clean_words = [w for w in words if w not in boring_words]

    # tokenizer = RegexpTokenizer(r'\w+')
    #TOD replace punctuateion in general
    clean_words = [cw.replace('.', '').replace('"', '').replace('\'', '') for cw in clean_words]
    return clean_words


def ndistance(s1, s2, foo):
    return foo(s1, s2)/min([len(s1), len(s2)])


def nlevenstein_root(s, s2, foo):
    """
    normed levenshtein metric
    :param s:
    :param s2:
    :param foo:
    :return:
    """
    if len(s) > len(s2):
        return nlevenstein_root(s2, s, foo)
    diff = len(s2) - len(s) + 1
    m = min([foo(s, s2[k:(k+len(s))]) for k in range(diff)])
    return m


def metric_to_words(m, foo, decision_thr=0.1, verbose=False):
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

    words_discrepancy_norm = np.sum(np.min(m, axis=0)**2/dim2)**0.5
    # e.g. ndistance(qw1, qw2, method=2)
    phrase_norm = foo(qw1, qw2)

    return phrase_norm, words_discrepancy_norm


def meta_distance(mw1, mw2, foo, foo_basic=None, sigma=0.5):
    # actually it is not symmetric (case when mw1 == mw2)
    if len(mw1) > len(mw2):
        return meta_distance(mw2, mw1, foo, foo_basic,sigma)
    if isinstance(mw1, str) and isinstance(mw2, str):
        if foo_basic:
            r = foo_basic(mw1, mw2)
        else:
            r = foo(mw1, mw2)
    else:
        m = np.array([[meta_distance(w1, w2, foo, foo_basic, sigma) for w1 in mw1] for w2 in mw2])
        da, db = metric_to_words(m, foo)
        r = ((da**2 + db**2)/2.)**0.5
    return r


def map_strings_to_phrases(strings, sep=',', asciize=True, cut_brackets=True, stopwords_languages=None):
    phrases_accum = []
    garbage_accum = []

    list_boring_words = [stopwords.words(language) for language in stopwords_languages]
    boring_words = [x for sublist in list_boring_words for x in sublist]

    if isinstance(strings[0], str):
        strings = zip(range(len(strings)), strings)
    for j, string in strings:
        phrases = split_string(string, sep, asciize, cut_brackets)
        tokenized_phrases = [tokenize_phrase(p, boring_words) for p in phrases]
        nonzero_len_phrases = list(filter(len, tokenized_phrases))
        if nonzero_len_phrases:
            phrases_accum.append((j, nonzero_len_phrases))
        else:
            garbage_accum.append((j, string))
    return phrases_accum, garbage_accum


def filter_targets(phrases_collection, targets=None, targets_back=None):
    phrases_accum = []
    garbage_accum = []
    # print(targets)
    for j, phrases in phrases_collection:
        masks = np.array([np.array([[t in y for y in p] for t in targets]).any() for p in phrases])
        shifted_mask = np.concatenate([[False], masks[:-1]])
        masks_back = np.array([np.array([[t == y for y in p] for t in targets_back]).any() for p in phrases])
        shifted_mask_back = np.concatenate([masks[1:], [False]])
        masks_rolled = masks | masks_back | shifted_mask_back
        # masks_rolled = masks | shifted_mask | masks_back | shifted_mask_back
        # masks_rolled = masks | shifted_mask
        # print(phrases)
        # print(masks, shifted_mask, masks_rolled)
        # print(masks, masks_rolled)
        phrases_rooted = [p for p, m in zip(phrases, masks_rolled) if m]

        if phrases_rooted:
            phrases_accum.append((j, phrases_rooted))
        else:
            garbage_accum.append((j, phrases))
    return phrases_accum, garbage_accum


def cluster_objects(objects, foo, foo_basic=None, simple_thr=0.3, max_it=None,
                    n_processes=1, verbose=False, debug=False):
    # objects is a list of tuples [(id, phrase)],
    # where id is an integer and phrase is a list of strings
    # debug True also returns id_cluster_dict with actual strings

    id_cluster_dict = {}
    k = 0
    objs = sorted(objects, key=lambda x: len(x[1]), reverse=True)
    with mp.Pool(n_processes) as p:
        while len(objs) > 1:
            cur_obj, objs = objs[0], objs[1:]

            objs_phrases = [x[1] for x in objs]

            if len(objs) > 500:
                    func = partial(meta_distance, mw2=cur_obj[1], foo=foo, foo_basic=foo_basic)
                    dists = np.array(p.map(func, objs_phrases))
            else:
                dists = np.array([meta_distance(cur_obj[1], a, foo, foo_basic) for j, a in objs])

            proximal_indices = np.where(dists < simple_thr)[0]
            str_rep = 'by simple'

            id_cluster_dict[k] = [cur_obj] + [objs[i] for i in proximal_indices]
            objs = [objs[i] for i in range(len(objs)) if i not in proximal_indices]

            size_dict = sum([len(x) for x in id_cluster_dict.values()])
            if verbose and len(proximal_indices) > 0:
                print('{0}: {1} objects clustered. {2} objects left. Total number of items {3}.'
                      .format(str_rep, len(proximal_indices) + 1, len(objs), len(objs) + size_dict))

            k += 1
            if max_it and k > max_it:
                break

    id_cluster_dict[k] = [objs[0]]

    index_phrase_index_cluster_map = {}

    for k, v in id_cluster_dict.items():
        for j, s in v:
            index_phrase_index_cluster_map[j] = k
    if debug:
        return index_phrase_index_cluster_map, id_cluster_dict
    else:
        return index_phrase_index_cluster_map


def reknit_big_small(dict_phrases, foo, size_cmp=3, thr=0.5, seed=0, verbose=False):
    rns = RandomState(seed)
    dnew = deepcopy(dict_phrases)

    keys = list(dnew.keys())

    keys_big = list(filter(lambda x: len(dnew[x]) > size_cmp, keys[:]))
    keys_sm = list(filter(lambda x: len(dnew[x]) <= size_cmp, keys[:]))
    if verbose:
        print(len(keys_sm), len(keys_big))
    for k_big in keys_big:
        size_k1 = min([size_cmp, len(dnew[k_big])])
        ii1 = rns.choice(len(dnew[k_big]), size_k1, False)
        items1 = [dnew[k_big][i][1] for i in ii1]
        # take size_cmp random elements from dd[k1]
        for k_sm in keys_sm:
            # take size_cmp random elements from dd[k1]
            size_k2 = min([size_cmp, len(dnew[k_sm])])
            ii2 = rns.choice(len(dnew[k_sm]), size_k2, False)
            items2 = [dnew[k_sm][i][1] for i in ii2]
            dists = np.array([[meta_distance(item1, item2, foo) for item2 in items2] for item1 in items1])
            min_dist = dists.min()
            if min_dist < thr:
                if verbose:
                    print('{0} (size {1}) -> {2} (size {3})'.format(k_sm, len(dnew[k_sm]), k_big, len(dnew[k_big])))
                dnew[k_big].extend(dnew[k_sm])
                del dnew[k_sm]
                keys_sm.pop(keys_sm.index(k_sm))
    return dnew


def prune_objects(dict_phrases, foo, size_cmp=3, thr=0.5, seed=0, verbose=False, debug=False):
    rns = RandomState(seed)
    dnew = {}
    keys_used = []
    keys = list(dict_phrases.keys())
    keys = sorted(keys, key=lambda x: len(dict_phrases[x]), reverse=True)

    for k in keys:
        dnew[k] = deepcopy(dict_phrases[k])
        cur_len = len(dnew[k])
        size_k1 = min([size_cmp + int(np.log2(cur_len)), cur_len])
        ii1 = rns.choice(len(dnew[k]), size_k1, False)
        items1 = [dnew[k][i][1] for i in ii1]
        # take size_cmp random elements from dd[k1]
        keys_loop = list((set(keys) - set(keys_used)) - {k})
        for q in keys_loop:
            # take size_cmp random elements from dd[k1]
            cur_len2 = len(dict_phrases[q])
            size_k2 = min([size_cmp + int(np.log2(cur_len2)), cur_len2])
            ii2 = rns.choice(len(dict_phrases[q]), size_k2, False)
            items2 = [dict_phrases[q][i][1] for i in ii2]
            dists = np.array([[meta_distance(item1, item2, foo) for item2 in items2] for item1 in items1])
            min_dist = dists.min()
            if min_dist < thr:
                if verbose:
                    print('{0} (size {1}) -> {2} (size {3})'.format(q, len(dict_phrases[q]), k, len(dnew[k])))
                dnew[k].extend(dict_phrases[q])
                keys_used.append(q)
                keys.pop(keys.index(q))
        keys_used.append(k)
        if verbose:
            print(sorted(list(dnew.keys())), len(dnew))

    index_phrase_index_cluster_map = {}

    for k, v in dnew.items():
        for j, s in v:
            index_phrase_index_cluster_map[j] = k
    if debug:
        return index_phrase_index_cluster_map, dnew
    else:
        return index_phrase_index_cluster_map
