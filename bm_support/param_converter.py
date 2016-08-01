from numpy import array, zeros, argsort


class params_converter(object):
    """
    very open parameter converter
    """

    def __init__(self, n):
        self.vars_dict = None

        self.guess_map = None
        self.names = None

        self.arr_guess = None
        self.arr_best = None

        self.raw_dict = None
        self.best_guess_dict = None

        self.n_modes = n

    def guess_arr_to_dict(self):
        vd = self.vars_dict
        ga = self.arr_guess
        gd = {}
        for k in vd:
            f = vd[k]['fwd_transform_func']
            if vd[k]['plate']:
                for j in range(self.n_modes):
                    kk = k + '_' + str(j) + vd[k]['suffix']
                    min_ = vd[k]['min']
                    max_ = vd[k]['max']
                    value = ga[self.guess_map[k], j]
                    gd[kk] = array([f(min_, max_, value)])
            else:
                kk = k + vd[k]['suffix']
                value = ga[self.guess_map[k]]
                gd[kk] = array(f(value))

        self.best_guess_dict = gd
        return gd

    def trans_dict_to_raw_dict(self):
        vd = self.vars_dict
        bgd = self.best_guess_dict
        rd = {}
        for k in vd:
            f = vd[k]['bwd_transform_func']
            if vd[k]['plate']:
                for j in range(self.n_modes):
                    kj = k + '_' + str(j)
                    kk = k + '_' + str(j) + vd[k]['suffix']
                    min_ = vd[k]['min']
                    max_ = vd[k]['max']
                    value = bgd[kk]
                    rd[kj] = array([f(min_, max_, value)])
            else:
                kk = k + vd[k]['suffix']
                value = bgd[kk]
                rd[k] = array(f(value))
        self.raw_dict = rd
        return rd

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