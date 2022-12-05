import numpy as np

from ab_testing import variance_reduction
from pytest import approx
from scipy.stats import norm, ttest_ind

def test_get_stratified_statistics():
    # Check both mean estimates have the same expected value
    size = 1000
    values = np.concatenate([
        norm(loc=0.3, scale=1).rvs(size=size),
        norm(loc=10.3, scale=1).rvs(size=size),
    ])
    strata = np.concatenate([np.array([0] * size), np.array([1] * size)])

    y_hat_strat, _ = variance_reduction.get_stratified_statistics(
        values=values,
        strata=strata,
        weights={0: .5, 1: .5}
    )

    assert y_hat_strat == approx(values.mean(), abs=.01)

def test_stratified_ttest():
    # TODO: add a static sample to ensure removing all randomness
    base = np.concatenate([
        norm(loc=0, scale=1).rvs(size=100),
        norm(loc=10, scale=1).rvs(size=100),
    ])
    strata_base = np.concatenate([np.array([0] * 100), np.array([1] * 100)])
    variant = np.concatenate([
        norm(loc=0.1, scale=1).rvs(size=100),
        norm(loc=10.1, scale=1).rvs(size=100),
    ])
    strata_variant = strata_base.copy()

    # Check consistency with usual t-test when providing only a single stratum
    p_unstratified, t_unstratified = variance_reduction.stratified_ttest(
        base=base,
        variant=variant,
        strata_base=base * 0,
        strata_variant=variant * 0,
        weights={0: 1},
    )

    t_check, p_check = ttest_ind(variant, base)
    # TODO: increase tolerance when figured out the correct degrees of freedom calculation in 'startified_ttest'
    assert (p_check, t_check) == approx((p_unstratified, t_unstratified), abs=0.001)

    # Check that using the correct strata decreases the p value
    p_stratified, t_stratified = variance_reduction.stratified_ttest(
        base=base,
        variant=variant,
        strata_base=strata_base,
        strata_variant=strata_variant,
        weights={0: .5, 1: .5},
    )

    print(f"\n\nInfo: p check: {p_check}, p stratified: {p_stratified}\n\n")

    assert p_stratified < p_check