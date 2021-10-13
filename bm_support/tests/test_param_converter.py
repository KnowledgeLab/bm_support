from ..param_converter import sort_dict_by_key
import unittest
from numpy import array, isclose


class TestConverter(unittest.TestCase):
    dd_test = {
        "mu_0": 9,
        "mu_1": 6,
        "mu_2": 12,
        "pi": array([1, 2, 15]),
        "t0_0": 14,
        "t0_1": 5,
        "t0_2": 3,
        "tau_0": 19,
        "tau_1": 4,
        "tau_2": 18,
    }

    dd_res = {
        "mu_0": 12,
        "mu_1": 6,
        "mu_2": 9,
        "pi": array([15, 2, 1]),
        "t0_0": 3,
        "t0_1": 5,
        "t0_2": 14,
        "tau_0": 18,
        "tau_1": 4,
        "tau_2": 19,
    }

    def test_sort_dict_by_sort_key(self):

        res = sort_dict_by_key(self.dd_test, "t0")
        print res.keys()
        if self.assertTrue(set(res.keys()) == set(self.dd_res.keys())):
            self.assertTrue(res == self.dd_res)
            # self.assertTrue(all([isclose(fwd_gu[k],
            #                              self.fwd_guess_ref[k], rtol=1e-4)
            # for k in fwd_gu.keys()]))


if __name__ == "__main__":
    unittest.main()
