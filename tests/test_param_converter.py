import bm_support.param_converter as bp
import unittest
import pymc3 as pm
from numpy import array, isclose
from bm_support.prob_constants import tau_min, tau_max
from bm_support.math_aux import sb_forward, sb_backward, \
    int_forward, int_backward


class TestConverter(unittest.TestCase):

    guess = array([[0.4396, 0.5604],
                   [42.8028, 53.8825],
                   [0.6622, 2.2004],
                   [1.016, 2.5141]])

    (tlow, thi) = (16.9931, 116.5180)

    (mu_min, mu_max) = (-3.0000, 5.0000)

    fwd_guess_ref = {
        'mu_0_interval_': array([-0.1693]),
        'mu_1_interval_': array([0.6193]),
        'pi_stickbreaking_': array([-0.2428]),
        't0_0_interval_': array([-1.0495]),
        't0_1_interval_': array([-0.5294]),
        'tau_0_interval_': array([-3.6722]),
        'tau_1_interval_': array([-2.712])
    }

    model_dict = {
        'mu': {'type': pm.Uniform, 'plate': True,
               'min': mu_min, 'max': mu_max},
        't0': {'type': pm.Uniform, 'plate': True,
               'min': tlow, 'max': thi},
        'tau': {'type': pm.Uniform, 'plate': True,
                'min': tau_min, 'max': tau_max},
        'pi': {'type': pm.Dirichlet, 'plate': False},
        #         'beta': {'type': pm.Normal, 'plate': True, 'dim': n_f_ext}
    }

    guess_list = ['pi', 't0', 'mu', 'tau']

    for k in model_dict:
        if model_dict[k]['type'] == pm.Dirichlet:
            model_dict[k]['fwd_transform_func'] = sb_forward
            model_dict[k]['bwd_transform_func'] = sb_backward
            model_dict[k]['suffix'] = '_stickbreaking_'
        elif model_dict[k]['type'] == pm.Uniform:
            model_dict[k]['fwd_transform_func'] = int_forward
            model_dict[k]['bwd_transform_func'] = int_backward
            model_dict[k]['suffix'] = '_interval_'

    def test_simple_converter(self):
        n_modes = 2
        pc = bp.ParamsConverter(n_modes)
        pc.arr_guess = self.guess
        pc.guess_map = {self.guess_list[i]: i for i in
                        range(len(self.guess_list))}

        pc.vars_dict = self.model_dict
        fwd_gu = pc.guess_arr_to_dict()
        if self.assertTrue(fwd_gu.keys() == self.fwd_guess_ref.keys()):
            self.assertTrue(all([isclose(fwd_gu[k],
                                         self.fwd_guess_ref[k], rtol=1e-4) for k in fwd_gu.keys()]))


if __name__ == '__main__':
    unittest.main()
