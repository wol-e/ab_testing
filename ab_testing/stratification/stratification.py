import numpy as np
import warnings

from scipy.stats import norm
from typing import Tuple, Dict

# TODO: move to using ab_test.ABTest class


def get_stratified_statistics(
        values: np.array,
        strata: np.array,
        weights: Dict,
) -> Tuple[float, float]:
    """
    :param values: array of values whose means you want to calculate with respect to strata
    :param strata: array of strata corresponding to values by index
    :param weights: dictionary mapping every stratum to its weight, e.g. {1: 2, 2: 4} means stratum 1 has weight 2 and
       stratum 2 has weight 4
    :return: tuple of floats, first value the stratified mean of values, second value the stratified variance
    """
    y_hat_strat = 0
    var_y_hat_strat = 0

    # TODO: move to proper Exceptions
    assert len(values) == len(strata), "lengths of observations and strata are not identical"

    _, strata_counts = np.unique(strata, return_counts=True)
    assert strata_counts.min() >= 2, "There need to be at least 2 observations per stratum"

    for stratum, weight in weights.items():
        obs = values[strata == stratum]
        y_hat_strat += obs.mean() * weight
        var_y_hat_strat += (weight / len(values)) * obs.var()

    return y_hat_strat, var_y_hat_strat
    

def stratified_ttest(
        base: np.array,
        variant: np.array,
        strata_base: np.array,
        strata_variant: np.array,
        weights: Dict,
        alternative: str = "two-sided",
) -> Tuple[float, float]:

    assert \
        set(strata_base) == set(strata_variant),\
        "The strata for base and variant need to have the same set of unique values"

    y_hat_strat_base, var_y_hat_strat_base = get_stratified_statistics(base, strata_base, weights)
    y_hat_strat_variant, var_y_hat_strat_variant = get_stratified_statistics(variant, strata_variant, weights)

    delta_strat = y_hat_strat_variant - y_hat_strat_base
    var_delta_strat = var_y_hat_strat_base + var_y_hat_strat_variant

    t_statistic = delta_strat / np.sqrt(var_delta_strat)

    # TODO: figure out the degrees of freedom for the t distribution. For now we'll use a normal distribution and
    #  raise a warning when we have less than 30 samples per group

    if min(len(base), len(variant)) <= 30:
        warnings.warn("Warning: Results might be inaccurate when there are not more than 30 observations in a group")

    approx_one_sided_p_val = 1 - norm(loc=0, scale=1).cdf(abs(t_statistic))

    if alternative == "two-sided":
        p_value = approx_one_sided_p_val * 2
    elif alternative == "one-sided":
        p_value = approx_one_sided_p_val
    else:
        raise ValueError("'alternative' must be wither 'two-sided' or 'one-sided'.")

    return p_value, t_statistic


if __name__ == "__main__":
    pass