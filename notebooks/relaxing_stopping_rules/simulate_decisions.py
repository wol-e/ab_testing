from typing import Type

from decision_rules import DecisionRule
from simulate import ABTestData


def simulate_decision_progression(ab_test_data: ABTestData, decision_rule: Type[DecisionRule], **kwargs_for_decision_rule):
    decision_progression = []
    for increment in range(ab_test_data.runtime):
        ab_test_data_until_increment = ABTestData(
            ab_test_data.samples_a[:increment + 1],
            ab_test_data.samples_b[:increment + 1]
        )
        decision = decision_rule(ab_test_data_until_increment, **kwargs_for_decision_rule).decision
        decision_progression.append(decision)
        if "STOP" in decision:
            break

    return {
        "DECISION": decision,
        "RUNTIME": len(decision_progression),
    }


if __name__ == "__main__":
    from decision_rules import (
        FixedHorizonDecisionRule,
        FixedHorizonDecisionRuleWithPeek,
        FixedHorizonDecisionRuleExtend,
    )
    from simulate import simulate_ab_test_data_binom, ABTestData
    fixed_horizon_decisions = []
    fixed_horizon_peeking_decisions = []

    data = simulate_ab_test_data_binom(
        n_samples_per_increment=100,
        n_increments=15,
        mean_a=.5,
        mean_b=.55
    )
    print(data.samples_a.mean(), data.samples_b.mean())
    fixed_horizon_decisions.append(
        simulate_decision_progression(
            data,
            FixedHorizonDecisionRule,
            **{
                "expected_runtime": 10,
                "alpha": .1,
                # "extension_interval": 2,
                # "extension_periods": 3
            },
        )
    )

    print(fixed_horizon_decisions)
