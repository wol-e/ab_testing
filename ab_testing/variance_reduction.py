import numpy as np

from scipy.stats import ttest_ind_from_stats, ttest_ind, norm
from typing import Tuple, Dict

def stratified_ttest(
        base: np.array,
        variant: np.array,
        strata_base: np.array,
        strata_variant: np.array,
        weights: Dict,
        alternative: str="two-sided",
) -> Tuple[float, float]:
    strata_by_weights_base = {}
    strata_by_weights_variant = {}

    # TODO: move to proper Exceptions
    assert\
        len(base) == len(strata_base) and len(variant) == len(strata_variant),\
        "lengths of observations and strata are not consistent"

    assert\
        set(strata_base) == set(strata_variant),\
        "The strata for base and variant need to have the same set of unique values"

    for strata in [strata_base, strata_variant]:
        _, strata_counts = np.unique(strata, return_counts=True)
        assert strata_counts.min() >= 2, "There need to be at least 2 observations per stratum"


    for stratum, weight in weights.items():
        strata_by_weights_base[weight] = base[strata_base == stratum]
        strata_by_weights_variant[weight] = variant[strata_variant == stratum]

    y_hat_strat_base = np.array([weight * obs.mean() for weight, obs in strata_by_weights_base.items()]).sum()
    y_hat_strat_variant = np.array([weight * obs.mean() for weight, obs in strata_by_weights_variant.items()]).sum()

    n_base, n_variant = len(base), len(variant)
    var_y_hat_strat_base = np.array(
        [(weight / n_base) * obs.var() for weight, obs in strata_by_weights_base.items()]
    ).sum()
    var_y_hat_strat_variant = np.array(
        [(weight / n_variant) * obs.var() for weight, obs in strata_by_weights_variant.items()]
    ).sum()

    delta_strat = y_hat_strat_variant - y_hat_strat_base
    var_delta_strat = var_y_hat_strat_base + var_y_hat_strat_variant

    t_statistic = delta_strat / np.sqrt(var_delta_strat)

    # TODO: figure out the degrees of freedom for the t distribution. For now we'll use a normal distribution and
    #  raise a warning when we have less than 30 samples per group
    if min(n_base, n_variant) <= 30:
        raise Warning("Warning: Results might be inaccurate when there are not more 30 observations in a group")

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