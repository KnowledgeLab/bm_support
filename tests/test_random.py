import bm_support.random as br
import unittest


class TestCollapse(unittest.TestCase):

    def test_generate_log_normal_mixture(self):
        data, pars = br.generate_log_normal_mixture()
        print(pars)
        # self.assertEquals(set(ret3[1].keys()), set(self.strings))


if __name__ == '__main__':
    unittest.main()
