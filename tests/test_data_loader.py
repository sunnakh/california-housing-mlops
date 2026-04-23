import numpy as np
import pandas as pd
import pytest

from common.logger import get_logger
from src.data.loader import FEATURE_COLS, TARGET_COL, load_california_housing, validate_data
from src.data.splitter import train_val_test_split

logger = get_logger(__name__)


class TestLoadCaliforniaHousing:
    """Tests for load_california_housing()."""

    def test_load_california_housing_returns_dataframe(self):
        df = load_california_housing()
        assert isinstance(df, pd.DataFrame)

    def test_dataframe_has_expected_shape(self):
        df = load_california_housing()
        assert df.shape[0] == 20640
        assert df.shape[1] == 9

    def test_dataframe_has_all_expected_columns(self):
        df = load_california_housing()
        expected = set(FEATURE_COLS + [TARGET_COL])
        assert expected == set(df.columns)

    def test_no_missing_values(self):
        df = load_california_housing()
        assert df.isnull().sum().sum() == 0

    def test_target_dtype_is_numeric(self):
        df = load_california_housing()
        assert pd.api.types.is_float_dtype(df[TARGET_COL])


class TestValidateData:
    """Tests for validate_data()."""

    def test_validate_passes_on_clean_data(self):
        df = load_california_housing()
        validate_data(df)  # must not raise

    def test_validate_catches_nulls(self):
        df = load_california_housing()
        df.loc[0, "MedInc"] = np.nan
        with pytest.raises(ValueError, match="nulls"):
            validate_data(df)

    def test_validate_catches_missing_columns(self):
        df = load_california_housing()
        df_missing = df.drop(columns=["MedInc"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_data(df_missing)


class TestTrainValTestSplit:
    """Tests for train_val_test_split()."""

    @pytest.fixture
    def sample_arrays(self):
        np.random.seed(42)
        x = np.random.randn(1000, 5)
        y = np.random.randn(1000)
        return x, y

    def test_split_total_size(self, sample_arrays):
        x, y = sample_arrays
        x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(x, y)
        assert len(x_train) + len(x_val) + len(x_test) == len(x)

    def test_split_raises_on_invalid_sizes(self, sample_arrays):
        x, y = sample_arrays
        with pytest.raises(ValueError):
            train_val_test_split(x, y, test_size=0.6, val_size=0.5)

    def test_x_y_aligned(self, sample_arrays):
        x, y = sample_arrays
        x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(x, y)
        assert len(x_train) == len(y_train)
        assert len(x_val) == len(y_val)
        assert len(x_test) == len(y_test)
