from numpy import array, zeros, argsort
import pymc3 as pm
from bm_support.math_aux import sb_forward, sb_backward, \
    int_forward, int_backward

cdict = {pm.Dirichlet: {}, pm.Uniform: {}}

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
    if set(d1.keys()) == set(d2.keys()):
        dcmp = {k: sum((d1[k] - d2[k])/d2[k]) if d1[k].shape
                else (d1[k] - d2[k])/d2[k] for k in d1.keys()}
        print sum(dcmp.values())
        return dcmp
    else:
        raise ValueError('dictionaries have incompatible keys: '
                         + ('{} '*len(d1.keys())).format(*d1.keys()) + ' and '
                         + ('{} '*len(d2.keys())).format(*d2.keys()))


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