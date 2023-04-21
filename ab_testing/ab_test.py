from numpy.typing import ArrayLike


class ABTest:
    def __init__(
            self,
            obs1: ArrayLike,
            obs2: ArrayLike,
            **kwargs,
    ):
        """

        :param obs1:
        :param obs2:
        """
        self.obs1 = obs1
        self.obs2 = obs2
        self.n1 = self.obs1.shape[0]
        self.n2 = self.obs2.shape[0]
        assert (self.n1 > 1, self.n2 > 1), "Both samples need at least 2 observations!"

    def report(self):
        pass

    def analyse(self):
        pass

    def update_obs(self, obs1=None, obs2=None, **kwargs):
        pass

class ABTestFromStats:
    def __init__(self):
        """
        Providing ABTest based on high levels statistics rather than raw data
        """
        raise NotImplementedError()
