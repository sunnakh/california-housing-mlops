from sklearn.linear_model import ElasticNet, Lasso, Ridge


def get_ridge(alpha: float = 1.0, random_state: int = 42) -> Ridge:

    return Ridge(alpha=alpha, random_state=random_state, fit_intercept=True)


def get_lasso(alpha: float = 0.1, random_state: int = 42) -> Lasso:

    return Lasso(alpha=alpha, random_state=random_state, fit_intercept=True, max_iter=10000)


def get_elastic_et(alpha: float = 0.1, l1_ratio: float = 0.5, random_state: int = 42) -> ElasticNet:

    return ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        random_state=random_state,
        fit_intercept=True,
        max_iter=10000,
    )
