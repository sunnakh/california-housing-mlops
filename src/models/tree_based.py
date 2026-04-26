from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


def get_decision_tree(**kwargs) -> DecisionTreeRegressor:

    defaults = {"random_state": 42}
    defaults.update(kwargs)
    return DecisionTreeRegressor(**defaults)


def get_random_forest(**kwargs) -> RandomForestRegressor:

    defaults = {
        "n_estimators": 100,
        "random_state": 42,
        "n_jobs": -1,
    }

    defaults.update(kwargs)
    return RandomForestRegressor(**defaults)


def get_extra_tree(**kwargs) -> ExtraTreesRegressor:

    defaults = {
        "n_estimators": 100,
        "random_state": 42,
        "n_jobs": -1,
    }

    defaults.update(kwargs)
    return ExtraTreesRegressor(**defaults)
