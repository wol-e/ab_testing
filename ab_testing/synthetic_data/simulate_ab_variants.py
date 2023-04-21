from scipy.stats import bernoulli


def generate_binomial_variants(
            n1: int = 100,
            n2: int = None,
            theta1: float = .5,
            theta2: float = None,
            random_state=None
    ):
    """
    :param n1:
    :param n2:
    :param theta1:
    :param theta2:
    :return:
    """
    if not n2:
        n2 = n1

    if not theta2:
        theta2 = theta1

    return (
            bernoulli(p=theta1).rvs(size=n1, random_state=random_state),
            bernoulli(p=theta2).rvs(size=n2, random_state=random_state)
    )


if __name__ == "__main__":
    # test scratch
    n1, n2 = 10, 20
    s1, s2 = generate_binomial_variants(n1=n1, n2=n2)
    assert s1.shape[0] == n1, s2.shape[0] == n2
