from ..random import generate_log_normal_mixture
from ..random import generate_logistic_y_from_bernoulli_x
from ..random import generate_log_normal_mixture_with_logistic
from ..random import generate_logistic_y_from_bernoulli_x_steplike_beta

from unittest import TestCase, main


class TestCollapse(TestCase):
    # TODO fix tests
    ref_par_lognormal_list = [
        "ps",
        "t0_1",
        "t0_0",
        "t0_2",
        "mu_2",
        "mu_1",
        "mu_0",
        "ns",
        "tau_2",
        "tau_0",
        "tau_1",
    ]
    ref_par_logistic_list = ["beta_1", "beta_0", "prior_p"]
    ref_par_lognormal_logistic_list = [
        "beta_1",
        "ps",
        "ns",
        "t0_1",
        "pp",
        "t0_0",
        "beta_0",
        "tau_1",
        "tau_0",
        "mu_1",
        "mu_0",
    ]

    ref_par_logistic_steplike_beta_list = [
        "bc_0",
        "bc_1",
        "bc_2",
        "bl_1",
        "bs_0",
        "bl_2",
        "bs_2",
        "bl_0",
        "br_2",
        "br_1",
        "br_0",
        "prior_p",
        "bs_1",
    ]

    seed = 17
    nf = 1
    n_modes_gen = 1
    ns = 500
    guess_types = ["random", "kmeans", "t_uniform"]

    def test_generate_log_normal_mixture(self):
        data, pars = generate_log_normal_mixture()
        self.assertEquals(set(pars.keys()), set(self.ref_par_lognormal_list))

    def test_generate_logistic(self):
        data, pars, c = generate_logistic_y_from_bernoulli_x()
        self.assertEquals(set(pars.keys()), set(self.ref_par_logistic_list))

    def test_log_normal_mixture_with_logistic(self):
        data, pars = generate_log_normal_mixture_with_logistic(2)
        # print(pars.keys())
        self.assertEquals(set(pars.keys()), set(self.ref_par_lognormal_logistic_list))

    def test_logistic_y_from_bernoulli_x_steplike_beta(self):
        # rns = RandomState(self.seed)

        data, pps_full = generate_log_normal_mixture(
            n_modes=self.n_modes_gen,
            seed=self.seed,
            n_samples=self.ns,
            t0_range=[0.0, 100.0],
            mu_range=[1.0, 4.0],
            tau_range=[4.0, 10.0],
            names={"ns": "ns", "t0": "t0", "mu": "mu", "tau": "tau", "ps": "pi"},
        )

        yd, pps_dict = generate_logistic_y_from_bernoulli_x_steplike_beta(data)
        self.assertEquals(
            set(pps_dict.keys()), set(self.ref_par_logistic_steplike_beta_list)
        )


if __name__ == "__main__":
    main()
