import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from src.features.feature_selector import (
    select_by_correlation,
    select_by_importance,
    select_by_variance,
)


class DummyModel:
    def fit(self, x, y):
        self.feature_importances_ = np.array([0.1, 0.5, 0.4])


class InvalidModel:
    def fit(self, x, y):
        pass


class TestFeatureSelector:
    def test_select_by_variance(self):

        x = np.array(
            [
                [1, 0, 1],
                [1, 1, 2],
                [1, 0, 3],
                [1, 1, 4],
            ]
        )

        result = select_by_variance(x=x)

        expected = np.array(
            [
                [0, 1],
                [1, 2],
                [0, 3],
                [1, 4],
            ]
        )
        assert result.shape == (4, 2)
        assert_array_equal(result, expected)

    def test_select_by_correlation(self):
        df_input = pd.DataFrame(
            {
                "f1": [1, 2, 3, 4],
                "f2": [1, 1, 1, 1],  # random low correlation
                "f3": [4, 3, 2, 1],  # negative correlation
                "target": [2, 4, 6, 8],  # target perfectly correlates with f1 and f3
            }
        )

        # with threshold 0.5 we should get f1 and f3
        result = select_by_correlation(df_input, target_col="target", threshold=0.5)
        assert "f1" in result
        assert "f3" in result
        assert "f2" not in result

    def test_select_by_importance(self):
        x = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ]
        )
        y = np.array([1, 2, 3])

        model = DummyModel()

        # Test basic functionality
        result = select_by_importance(x, y, model=model, top_n=2)

        # Expected indices: 1 (0.5), 2 (0.4)
        expected = np.array(
            [
                [2, 3],
                [5, 6],
                [8, 9],
            ]
        )
        assert_array_equal(result, expected)

        # Test attribute error when model has no feature_importances_
        invalid_model = InvalidModel()
        with pytest.raises(AttributeError, match="does not have feature_importances_"):
            select_by_importance(x, y, model=invalid_model)

        # Test value error for mismatching feature names length
        with pytest.raises(ValueError, match="feature_names length must match X.shape"):
            select_by_importance(x, y, model=model, feature_names=["f1", "f2"])
