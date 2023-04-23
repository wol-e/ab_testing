import numpy as np

from ab_testing.synthetic_data.simulate_ab_variants import generate_binomial_variants
from ab_testing.ab_test import ABTest
from typing import List, Dict
# from scipy.stats import ttest_ind
# from statsmodels.stats.weightstats import CompareMeans # ztest_ind
from scipy.stats import zscore, norm

class SPRT(ABTest):
    """
    Based on https://en.wikipedia.org/wiki/Sequential_probability_ratio_test

    Idea: For a bernoulli metric we want to AB test, e.g. does a user convert, and we want to estimate if the
    conversion rate in obs1 is different from obs2 we can use SPRT as follows:
    Let p1 and p2 be the conversion rates of the two populations and let d be our difference in means
    we want to detect in the populations. When collecting data, we can after every batch of collected data,
    e.g. every day or every week, calculate the means and the difference in means for the badges. Now we can
    perform an SPRT test with H_0 = 0 and H_1 = d: We calculate the likelihood of our observed means for
    both H_0 and H_1 and then calculate the log likelihood ratio.

    Note: By design this is a one-sided test I guess, so we might need to do some thinking
    """
    def __init__(
            self,
            d,
            beta: float = .8,
            looks: List = None,
            kwargs_for_ttest: Dict = {},
            **kwargs
    ):
        """

        :param d:
        :param beta:
        :param looks:
        :param kwargs_for_ttest:
        :param kwargs:
        """
        self.d = d
        self.beta = beta
        self.looks = looks
        self.kwargs_for_ttest = kwargs_for_ttest
        if not self.looks:
            self.looks = []  # if no looks are defined we look at all data once
        super().__init__(**kwargs)
        if self.n1 != self.n2:
            raise ValueError("SPRT test with unequal sample sizes is not yet implemented!")  # TODO
        self.looks = sorted(list(set(self.looks + [self.n1])))
        self.obs1_slices = []
        self.obs2_slices = []
        self.stats = {}

    def analyse(self):
        slices = list(zip([0] + self.looks[:self.n1], self.looks))
        self.obs1_slices = [self.obs1[e[0]:e[1]] for e in slices]
        self.obs2_slices = [self.obs2[e[0]:e[1]] for e in slices]
        means = [[_slice.mean() for _slice in slices] for slices in [self.obs1_slices, self.obs2_slices]]
        self.stats["slice_means"] = means
        stds = [[_slice.std() for _slice in slices] for slices in [self.obs1_slices, self.obs2_slices]]
        self.stats["slice_stds"] = stds

        zscores1 = [zscore(obs) for obs in self.obs1_slices]
        zscores2 = [zscore(obs) for obs in self.obs2_slices]

        likelihoods1 = [norm.pdf(z) for z in zscores1]
        likelihoods2 = [norm.pdf(z) for z in zscores2]

        log_likelihood_ratios = [np.log(likelihoods2[i] / l) for i, l in enumerate(likelihoods1)]

        assert False

    def report(self):
        if not self.stats:
            self.analyse()
        return f"AB Test report for Bonferroni corrected Sequential Analysis:\n\n" +\
        f"Based on {self.n1} observations per group and {len(self.looks)} interim analyses" +\
            f" with an alpha level of {self.alpha} we obtain bla bla bla..."


if __name__ == "__main__":
    obs1, obs2 = generate_binomial_variants(n1=100, n2=100, theta1=.5, theta2=.8)
    test = SPRT(obs1=obs1, obs2=obs2, d=.1, looks=[20, 40, 60, 80, 100])
    test.analyse()
    end = "Hurray"
