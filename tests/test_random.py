import bm_support.random as br
import unittest


class TestCollapse(unittest.TestCase):

    ref_par_lognormal_list = ['ps', 't0_1', 't0_0', 't0_2', 'mu_2',
                              'mu_1', 'mu_0', 'ns', 'tau_2', 'tau_0', 'tau_1']
    ref_par_logistic_list = ['beta_1', 'beta_0']
    ref_par_lognormal_logistic_list = ['beta_1', 'ps', 'ns', 't0_1',
                                       't0_0', 'beta_0', 'tau_1', 'tau_0', 'mu_1', 'mu_0']

    def test_generate_log_normal_mixture(self):
        data, pars = br.generate_log_normal_mixture()
        self.assertEquals(pars.keys(), self.ref_par_lognormal_list)

    def test_generate_logistic(self):
        data, pars, c, d = br.generate_logistic()
        self.assertEquals(pars.keys(), self.ref_par_logistic_list)

    def test_log_normal_mixture_with_logistic(self):
        data, pars, xb = br.generate_log_normal_mixture_with_logistic(2)
        # print(pars.keys())
        self.assertEquals(pars.keys(), self.ref_par_lognormal_logistic_list)

if __name__ == '__main__':
    unittest.main()
