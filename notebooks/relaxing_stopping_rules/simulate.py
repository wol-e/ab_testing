from scipy.stats import bernoulli
from typing import Type

class ABTestData:
    def __init__(self, samples_a, samples_b):
        self.runtime = len(samples_a)
        assert self.runtime == len(samples_b),\
            "Error: samples_control and samples_variant must have the same length"
        self.samples_a = samples_a
        self.samples_b = samples_b


def simulate_ab_test_data_binom(n_samples_per_increment, n_increments, mean_a, mean_b):
    samples_a = bernoulli(p=mean_a).rvs(n_increments * n_samples_per_increment)\
        .reshape(n_increments, n_samples_per_increment)
    samples_b = bernoulli(p=mean_b).rvs(n_increments * n_samples_per_increment) \
        .reshape(n_increments, n_samples_per_increment)

    return ABTestData(samples_a=samples_a, samples_b=samples_b)
