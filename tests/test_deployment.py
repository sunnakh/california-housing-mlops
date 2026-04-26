import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

# ── Unit tests for schemas ────────────────────────────────────────────────────

from src.deployment.schemas import (
    BatchPredictionRequest,
    HousingFeatures,
)

VALID_FEATURES = {
    "MedInc": 3.5,
    "HouseAge": 28.0,
    "AveRooms": 5.2,
    "AveBedrms": 1.1,
    "Population": 1200.0,
    "AveOccup": 3.0,
    "Latitude": 37.88,
    "Longitude": -122.23,
}


class TestHousingFeaturesSchema:
    def test_valid_features_parse(self) -> None:
        f = HousingFeatures(**VALID_FEATURES)
        assert f.MedInc == 3.5
        assert f.Latitude == 37.88

    def test_negative_medinc_raises(self) -> None:
        from pydantic import ValidationError

        bad = {**VALID_FEATURES, "MedInc": -1.0}
        with pytest.raises(ValidationError):
            HousingFeatures(**bad)

    def test_latitude_out_of_range_raises(self) -> None:
        from pydantic import ValidationError

        bad = {**VALID_FEATURES, "Latitude": 50.0}  # outside California
        with pytest.raises(ValidationError):
            HousingFeatures(**bad)

    def test_longitude_out_of_range_raises(self) -> None:
        from pydantic import ValidationError

        bad = {**VALID_FEATURES, "Longitude": -100.0}  # east of California
        with pytest.raises(ValidationError):
            HousingFeatures(**bad)

    def test_ave_rooms_less_than_one_raises(self) -> None:
        from pydantic import ValidationError

        bad = {**VALID_FEATURES, "AveRooms": 0.5}
        with pytest.raises(ValidationError):
            HousingFeatures(**bad)

    def test_model_dump_returns_dict(self) -> None:
        f = HousingFeatures(**VALID_FEATURES)
        d = f.model_dump()
        assert isinstance(d, dict)
        assert set(d.keys()) == set(VALID_FEATURES.keys())


class TestBatchRequest:
    def test_single_instance_accepted(self) -> None:
        req = BatchPredictionRequest(instances=[HousingFeatures(**VALID_FEATURES)])
        assert len(req.instances) == 1

    def test_empty_instances_raises(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            BatchPredictionRequest(instances=[])

    def test_over_max_instances_raises(self) -> None:
        from pydantic import ValidationError

        instances = [HousingFeatures(**VALID_FEATURES)] * 1001
        with pytest.raises(ValidationError):
            BatchPredictionRequest(instances=instances)


# ── Unit tests for ModelServer ────────────────────────────────────────────────

from src.deployment.model_server import ModelServer


class TestModelServer:
    def test_predict_before_load_raises(self) -> None:
        server = ModelServer(
            model_path="nonexistent.joblib",
            preprocessor_path="nonexistent.joblib",
        )
        with pytest.raises(RuntimeError, match="load"):
            server.predict([VALID_FEATURES])

    def test_empty_instances_raises(self) -> None:
        server = ModelServer(
            model_path="nonexistent.joblib",
            preprocessor_path="nonexistent.joblib",
        )
        server._is_loaded = True  # bypass load check
        server._model = MagicMock()
        server._preprocessor = MagicMock()
        with pytest.raises(ValueError, match="empty"):
            server.predict([])

    def test_predict_calls_model(self) -> None:
        """Mock the model and preprocessor to test the prediction flow."""
        server = ModelServer(
            model_path="m.joblib",
            preprocessor_path="p.joblib",
        )
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([2.5])
        mock_preprocessor = MagicMock()
        mock_preprocessor.transform.return_value = np.zeros((1, 11))  # 8 base + 3 engineered

        server._model = mock_model
        server._preprocessor = mock_preprocessor
        server._is_loaded = True

        result = server.predict([VALID_FEATURES])
        assert len(result) == 1
        assert mock_model.predict.called
        assert mock_preprocessor.transform.called


# ── Integration tests for FastAPI endpoints (no server process) ───────────────


@pytest.fixture
def mock_server():
    """Return a ModelServer with a mock model that returns 2.5 for any input."""
    server = ModelServer(
        model_path="m.joblib",
        preprocessor_path="p.joblib",
        model_name="test_model",
        model_version="0.0.1",
    )
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([2.5])
    mock_preprocessor = MagicMock()
    mock_preprocessor.transform.side_effect = lambda x: x
    server._model = mock_model
    server._preprocessor = mock_preprocessor
    server._is_loaded = True
    return server


@pytest.mark.asyncio
async def test_health_endpoint_ok(mock_server) -> None:
    """Health endpoint returns 200 and status ok when model is loaded."""
    from httpx import ASGITransport, AsyncClient

    from src.deployment import app as app_module

    with patch.object(app_module, "_model_server", mock_server):
        async with AsyncClient(
            transport=ASGITransport(app=app_module.app), base_url="http://test"
        ) as client:
            response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


@pytest.mark.asyncio
async def test_predict_endpoint_returns_prediction(mock_server) -> None:
    """POST /predict returns a valid prediction response."""
    from httpx import ASGITransport, AsyncClient

    from src.deployment import app as app_module

    with patch.object(app_module, "_model_server", mock_server):
        async with AsyncClient(
            transport=ASGITransport(app=app_module.app), base_url="http://test"
        ) as client:
            response = await client.post("/predict", json=VALID_FEATURES)

    assert response.status_code == 200
    data = response.json()
    assert "predicted_value" in data
    assert "predicted_value_usd" in data
    assert data["predicted_value_usd"] == pytest.approx(data["predicted_value"] * 100_000, rel=0.01)


@pytest.mark.asyncio
async def test_predict_invalid_input_returns_422(mock_server) -> None:
    """POST /predict with invalid data returns HTTP 422."""
    from httpx import ASGITransport, AsyncClient

    from src.deployment import app as app_module

    bad_input = {**VALID_FEATURES, "MedInc": -5.0}  # negative income
    with patch.object(app_module, "_model_server", mock_server):
        async with AsyncClient(
            transport=ASGITransport(app=app_module.app), base_url="http://test"
        ) as client:
            response = await client.post("/predict", json=bad_input)

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_batch_predict_endpoint(mock_server) -> None:
    """POST /predict/batch returns a list of predictions."""
    from httpx import ASGITransport, AsyncClient

    from src.deployment import app as app_module

    mock_server._model.predict.return_value = np.array([2.5, 3.1])
    batch_body = {"instances": [VALID_FEATURES, VALID_FEATURES]}

    with patch.object(app_module, "_model_server", mock_server):
        async with AsyncClient(
            transport=ASGITransport(app=app_module.app), base_url="http://test"
        ) as client:
            response = await client.post("/predict/batch", json=batch_body)

    assert response.status_code == 200
    data = response.json()
    assert data["n_samples"] == 2
    assert len(data["predictions"]) == 2
