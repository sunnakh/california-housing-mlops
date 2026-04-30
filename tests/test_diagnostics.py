from unittest.mock import patch

import numpy as np

from src.evaluation.diagnostics import (
    check_heterosedasticity,
    plot_prediction_vs_actual,
    plot_residuals,
)


class TestDiagnostics:
    @patch("src.evaluation.diagnostics._save_or_show")
    @patch("src.evaluation.diagnostics.logger")
    def test_plot_residuals(self, mock_logger, mock_save_or_show):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8])

        plot_residuals(y_true, y_pred)

        assert mock_save_or_show.called

        assert mock_logger.info.called
        log_message = mock_logger.info.call_args[0][0]
        assert "Residual stats" in log_message

    @patch("src.evaluation.diagnostics._save_or_show")
    def test_plot_prediction_vs_actual(self, mock_save_or_show):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8])

        plot_prediction_vs_actual(y_true, y_pred)

        assert mock_save_or_show.called

    @patch("src.evaluation.diagnostics.smd.het_breuschpagan")
    @patch("src.evaluation.diagnostics.logger")
    def test_check_heterosedasticity(self, mock_logger, mock_bp_test, capsys):
        mock_bp_test.return_value = (10.5, 0.01, 15.0, 0.005)

        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8])

        check_heterosedasticity(y_true, y_pred)

        assert mock_bp_test.called

        assert mock_logger.info.called
        log_message = mock_logger.info.call_args[0][0]
        assert "Breusch-Pagan test: LM=10.5" in log_message

        captured = capsys.readouterr()
        assert "Breusch-Pagan Heteroscedasticity Test:" in captured.out
        assert "Evidence of heteroscedasticity (p < 0.05)" in captured.out

        mock_bp_test.return_value = (2.1, 0.15, 3.0, 0.1)
        check_heterosedasticity(y_true, y_pred)

        captured = capsys.readouterr()
        assert "No significant evidence of heteroscedasticity" in captured.out
