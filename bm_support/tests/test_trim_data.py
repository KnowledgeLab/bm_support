from unittest import TestCase, main
from numpy.random import RandomState
from ..posterior_tools import trim_data


class TestTrim_data(TestCase):
    rns = RandomState(17)
    data = rns.normal(size=200)
    a_ref = -1.3764
    b_ref = 1.3216

    def test_trim_data(self):
        a, b = trim_data(data=self.data, n_bins=10)
        diff = ((self.a_ref - a)**2 + (self.b_ref - b)**2)**0.5
        self.assertAlmostEqual(diff, 0.0, 3)

if __name__ == '__main__':
    main()
