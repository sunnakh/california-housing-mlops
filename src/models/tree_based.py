from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor


def get_decision_tree(**kwargs) -> DecisionTreeRegressor:

    defaults = {"random_state": 42}
    defaults.update(kwargs=kwargs)
    return DecisionTreeRegressor(**defaults)


def get_random_forest(**kwargs) -> RandomForestRegressor:

    defaults = {
        "n_estimators": 100,
        "random_state": 42,
        "n_jobs": -1,
    }

    defaults.update(kwargs=kwargs)
    return RandomForestRegressor(**defaults)


def get_extra_tree(**kwargs) -> ExtraTreeRegressor:

    defaults = {
        "n_estimators": 100,
        "random_state": 42,
        "n_jobs": -1,
    }

    defaults.update(kwargs=kwargs)
    return ExtraTreeRegressor(**defaults)
