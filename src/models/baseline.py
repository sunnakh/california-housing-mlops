"""
Baseline regression models.

Baselines establish the floor that all subsequent models must beat.
The DummyRegressor (predicts mean) is the absolute minimum bar.
A linear regression that cannot beat DummyRegressor has a bug.

Both functions return unfitted estimators following the sklearn convention —
the caller is responsible for fitting.
"""

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression


def get_dummy_regressor() -> DummyRegressor:
    """Return an unfitted DummyRegressor that predicts the mean of the training set."""

    return DummyRegressor(strategy="mean")


def get_linear_regression() -> LinearRegression:
    """Return an unfitted ordinary least squares LinearRegression estimator."""

    return LinearRegression(fit_intercept=True, n_jobs=-1)
