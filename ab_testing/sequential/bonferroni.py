from ab_testing.synthetic_data.simulate_ab_variants import generate_binomial_variants
from ab_testing.ab_test import ABTest
from typing import List, Dict
from scipy.stats import ttest_ind

class Bonferroni(ABTest):
    def __init__(
            self,
            alpha: float = 0.05,
            looks: List = None,
            kwargs_for_ttest: Dict = {},
            **kwargs
    ):
        self.alpha = 0.05
        self.looks = looks
        self.kwargs_for_ttest = kwargs_for_ttest
        if not self.looks:
            self.looks = []  # if no looks are defined we look at all data once
        super().__init__(**kwargs)
        if self.n1 != self.n2:
            raise ValueError("Bonferroni sequential testing with unequal sample sizes is not yet implemented!")  # TODO
        self.looks = sorted(list(set(self.looks + [self.n1])))
        self.bonferroni_factor = len(self.looks)
        self.corrected_alpha = self.alpha / self.bonferroni_factor
        self.stats = {}

    def analyse(self):
        means = [[obs[:look].mean() for look in self.looks] for obs in [self.obs1, self.obs2]]
        self.stats["means"] = means
        stds = [[obs[:look].std() for look in self.looks] for obs in [self.obs1, self.obs2]]
        self.stats["stds"] = stds
        p_vals_t_vals = [
            ttest_ind(
                a=self.obs1[:look],
                b=self.obs2[:look],
                **self.kwargs_for_ttest
            ) for look in self.looks
        ]
        self.stats["p_values_raw"] = [r.pvalue for r in p_vals_t_vals]
        self.stats["p_values_corrected"] = [min(p * self.bonferroni_factor, 1) for p in self.stats["p_values_raw"]]
        self.stats["t_values"] = [r.statistic for r in p_vals_t_vals]

    def report(self):
        if not self.stats:
            self.analyse()
        return f"AB Test report for Bonferroni corrected Sequential Analysis:\n\n" +\
        f"Based on {self.n1} observations per group and {len(self.looks)} interim analyses" +\
            f" with an alpha level of {self.alpha} we obtain bla bla bla..."


if __name__ == "__main__":
    obs1, obs2 = generate_binomial_variants(n1=100, n2=100, theta1=.5, theta2=.8)
    test = Bonferroni(obs1=obs1, obs2=obs2, looks=[10, 50, 75, 100])
    test.analyse()
    end = "Hurray"
