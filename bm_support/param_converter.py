from numpy import array, zeros, argsort, sqrt
import pymc3 as pm
from bm_support.math_aux import sb_forward, sb_backward, \
    int_forward, int_backward, logodds_forward, logodds_backward

cdict = {pm.Dirichlet: {}, pm.Uniform: {}, pm.Beta: {}}

cdict[pm.Beta]['fwd_transform_func'] = logodds_forward
cdict[pm.Beta]['bwd_transform_func'] = logodds_backward
cdict[pm.Beta]['suffix'] = '_logodds_'

cdict[pm.Dirichlet]['fwd_transform_func'] = sb_forward
cdict[pm.Dirichlet]['bwd_transform_func'] = sb_backward
cdict[pm.Dirichlet]['suffix'] = '_stickbreaking_'

cdict[pm.Uniform]['fwd_transform_func'] = int_forward
cdict[pm.Uniform]['bwd_transform_func'] = int_backward
cdict[pm.Uniform]['suffix'] = '_interval_'


def map_parameters(left_dict, model_dict, ranges, forward):
    """

    :param left_dict:
    :param model_dict:
    :param conv_dict:
    :param ranges:
    :param forward:
    :return:
    """
# left_dict -> right_dict
# raw -> trans if forward, else trans -> raw
    md = model_dict
#     left_dict
    ld = left_dict
    rd = {}
    cd = cdict

    for k in ld:
        kk = k.split('_')[0]
#       distr_type
        if kk in md.keys():
            dt = md[kk]['type']
            if dt in cd.keys():
                suffix = cd[dt]['suffix']
                if forward:
                    f = cd[dt]['fwd_transform_func']
                    krd = k + suffix
                else:
                    f = cd[dt]['bwd_transform_func']
                    if k.endswith(suffix):
                        krd = k[:-len(suffix)]
                    else:
                        raise ValueError('suffix {} should be present in key {}'.format(suffix, k))
                if kk in ranges:
                    rd[krd] = array(f(ranges[kk][0], ranges[kk][1], ld[k]))
                else:
                    rd[krd] = array(f(ld[k]))
            else:
                rd[k] = ld[k]
    return rd


def dict_cmp(d1, d2):
    """
    produce the numeric relative differences
    of two dictionaries with identical keys,
    providing the square root of the sum of the squares for the array
    :param d1: first dict
    :param d2: second dict
    :return: dict with rel diffs
    """
    if set(d1.keys()) == set(d2.keys()):
        dcmp = {k: sqrt(sum(((d1[k] - d2[k])/d2[k])**2)) if d1[k].shape
                else (d1[k] - d2[k])/d2[k] for k in d1.keys()}
        print sum(dcmp.values())
        return dcmp
    else:
        raise ValueError('dictionaries have incompatible keys: '
                         + ('{} '*len(d1.keys())).format(*d1.keys()) + ' and '
                         + ('{} '*len(d2.keys())).format(*d2.keys()))


def sort_dict_by_key(input_dict, sort_key):
    # enum_keys = [k for k in input_dict.keys() if len(k.split('_')) > 1]
    # base_keys = set([k.split('_')[0] for k in enum_keys])
    # other_keys = list(set(input_dict.keys()) - set(enum_keys))
    # enums = sorted(list(set([int(k.split('_')[-1]) for k in enum_keys])))

    enums, base_keys, other_keys, enum_keys = break_dict_to_enums_and_notenums(input_dict)

    if sort_key in base_keys:
        out_dict = {}
        to_sort_list = [input_dict[sort_key + '_' + str(k)] for k in enums]
        sorting_order = sorted(range(len(to_sort_list)), key=lambda i: to_sort_list[i])
        sorting_dict = dict(zip(sorting_order, range(len(to_sort_list))))
        for k in base_keys:
            upd = {k + '_' + str(_to): input_dict[k + '_' + str(_from)] for _from, _to in sorting_dict.iteritems()}
            out_dict.update(upd )
        out_dict.update({k: input_dict[k][sorting_order] for k in other_keys})
        return out_dict
    else:
        raise ValueError('sort_key arg should be in the input_dict keys()')


def break_dict_to_enums_and_notenums(input_dict):
    enum_keys = [k for k in input_dict.keys() if len(k.split('_')) > 1]
    base_keys = set([k.split('_')[0] for k in enum_keys])
    other_keys = list(set(input_dict.keys()) - set(enum_keys))
    enums = sorted(list(set([int(k.split('_')[-1]) for k in enum_keys])))
    return enums, base_keys, other_keys, enum_keys


def merge_two_dicts(x, y):
    """
    Given two dicts, merge them into a new dict as a shallow copy
    """
    z = x.copy()
    z.update(y)
    return z


def dict_to_list(input_dict):
    enums, base_keys, other_keys, _ = break_dict_to_enums_and_notenums(input_dict)

    par_dicts = [{k: input_dict[k + '_' + str(i)] for k in base_keys} for i in enums]
    par_dicts2 = [{k: input_dict[k][i] for k in other_keys} for i in enums]
    p_dicts = [merge_two_dicts(d1, d2) for d1, d2 in zip(par_dicts, par_dicts2)]
    return p_dicts


def raw_dict_to_arr(self, key_to_order='t0'):
    gm = self.guess_map
    rd = self.raw_dict
    self.arr_best = zeros(self.arr_guess.shape)
    for k in gm.keys():
        kkeys = [q for q in rd.keys() if k in q]
        for kk in kkeys:
            words = kk.split('_')
            if len(words) > 1:
                self.arr_best[gm[k], int(words[1])] = rd[kk]
            else:
                self.arr_best[gm[k]] = rd[kk]

    ord_key = argsort(self.arr_best[gm[key_to_order]])
    self.arr_best = self.arr_best[:, ord_key]
    return self.arr_best