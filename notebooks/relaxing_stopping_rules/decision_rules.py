from scipy.stats import ttest_ind

from simulate import ABTestData


class Decision:
    def __init__(self, decision):
        assert decision in ["CONTINUE", "STOP REJECT NULL", "STOP INCONCLUSIVE", "STOP ACCEPT NULL"]
        self.value = decision


class DecisionRule:
    """

    """
    def __init__(self, ab_test_data: ABTestData, *args, **kwargs):
        self.ab_test_data = ab_test_data
        self.test_runtime = ab_test_data.runtime
        self.decision = None


class FixedHorizonDecisionRule(DecisionRule):
    def __init__(self, ab_test_data: ABTestData, expected_runtime: int, alpha: float, **kwargs_for_ttest):
        super().__init__(ab_test_data)
        assert expected_runtime > 0, "Error: expected_runtime must be positive"
        assert expected_runtime >= ab_test_data.runtime,\
            "Error: the runtime of the test to be analysed is longer than the expected runtime"
        self.alpha = alpha
        if expected_runtime > ab_test_data.runtime:
            self.decision = Decision("CONTINUE")
            self.p = None
            self.t = None
        else:  # runtime equals expected runtime
            self.t, self.p = ttest_ind(
                ab_test_data.samples_a.reshape(-1),
                ab_test_data.samples_b.reshape(-1),
                **kwargs_for_ttest
            )
            self.decision = Decision("STOP REJECT NULL") if self.p <= alpha else Decision("STOP INCONCLUSIVE")

        self.decision = self.decision.value


class FixedHorizonDecisionRuleContinuousPeek(DecisionRule):
    def __init__(self, ab_test_data: ABTestData, expected_runtime: int, alpha: float, **kwargs_for_ttest):
        super().__init__(ab_test_data)
        assert expected_runtime > 0, "Error: expected_runtime must be positive"
        assert expected_runtime >= ab_test_data.runtime,\
            "Error: the runtime of the test to be analysed is longer than the expected runtime"
        self.alpha = alpha
        self.t, self.p = ttest_ind(
            ab_test_data.samples_a.reshape(-1),
            ab_test_data.samples_b.reshape(-1),
            **kwargs_for_ttest
        )
        if self.p <= self.alpha:
            self.decision = Decision("STOP REJECT NULL")
        elif expected_runtime == ab_test_data.runtime:
            self.decision = Decision("STOP INCONCLUSIVE")
        else:
            self.decision = Decision("CONTINUE")

        self.decision = self.decision.value


class FixedHorizonDecisionRuleWeeklyPeek(DecisionRule):
    pass
    # TODO


class FixedHorizonDecisionRuleExtend(DecisionRule):
    pass
# TODO
