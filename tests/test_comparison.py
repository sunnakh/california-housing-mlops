import pandas as pd
import pytest

from src.evaluation.comparison import ModelComparison


class TestComparison:
    def test_get_best_model(self):

        model_comparison = ModelComparison()
        with pytest.raises(ValueError, match="No results registered"):
            model_comparison.get_best_model()

    def test_add_results(self):

        model_comparison = ModelComparison()

        model_comparison.add_results(
            model_name="baseline", metrics_dict={"rmse": 0.2, "r2": 0.8}, train_time=0.2
        )

        model_comparison.add_results(
            model_name="ridge", metrics_dict={"rmse": 0.3, "r2": 0.7}, train_time=0.1
        )

        assert model_comparison.get_best_model("rmse") == "baseline"

    def test_store_best_params(self):
        model_comparison = ModelComparison()

        baseline_params = {"rmse": [0.2, 0.3, 0.4], "r2": [0.8, 0.9, 1.0]}
        updated_baseline_params = {"rmse": [0.15, 0.2], "r2": [0.91, 0.93]}

        model_comparison.store_best_params(model_name="baseline", params=baseline_params)
        assert model_comparison._best_params["baseline"] == baseline_params

        model_comparison.store_best_params(model_name="baseline", params=updated_baseline_params)
        assert model_comparison._best_params["baseline"] == updated_baseline_params

    def test_to_dataframe(self):

        obj = ModelComparison()
        obj._results = {"baseline": {"rmse": 0.2, "r2": 0.8}, "ridge": {"rmse": 0.3, "r2": 0.7}}

        df = obj.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)
        assert df.index.name == "Model"
