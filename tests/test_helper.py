import time
from unittest.mock import patch

import pytest

from src.utils.helpers import load_model, save_model, timer


class TestHelpers:
    @patch("src.utils.helpers.logger")
    def test_timer_context_manager(self, mock_logger):

        with timer(label="Test execution"):
            time.sleep(0.01)

        assert mock_logger.info.called
        log_message = mock_logger.info.call_args[0][0]
        assert "Test execution completed in" in log_message

    def test_save_and_load_model(self, tmp_path):
        dummy_model = {"model_name": "TestModel", "weights": [1, 2, 3]}
        model_path = tmp_path / "subdir" / "test_model.pkl"

        # Save model
        save_model(dummy_model, model_path)
        assert model_path.exists()
        assert model_path.is_file()

        # Load model
        loaded_model = load_model(model_path)
        assert loaded_model == dummy_model

    def test_load_model_file_not_found(self, tmp_path):
        non_existent_path = tmp_path / "non_existent_model.pkl"

        with pytest.raises(FileNotFoundError, match="Model file not found"):
            load_model(non_existent_path)
