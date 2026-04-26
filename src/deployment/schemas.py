"""
Pydantic request and response schemas for the regression model REST API.

Pydantic provides automatic validation, serialization, and OpenAPI
documentation generation. Each field has a description that appears in
the /docs Swagger UI.
"""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HousingFeatures(BaseModel):
    """
    Input features for a single California Housing prediction request.

    All features correspond to one census block group. Values outside the
    training range are accepted but may produce less reliable predictions.
    """

    MedInc: float = Field(
        ...,
        ge=0.0,
        description="Median income in block group (tens of thousands of USD). Range: ~0.5–15.",
    )
    HouseAge: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Median house age in the block group (years). Range: 1–52.",
    )
    AveRooms: float = Field(
        ...,
        ge=0.0,
        description="Average number of rooms per household. Range: ~1–150.",
    )
    AveBedrms: float = Field(
        ...,
        ge=0.0,
        description="Average number of bedrooms per household. Range: ~0.5–35.",
    )
    Population: float = Field(
        ...,
        ge=0.0,
        description="Block group population. Range: 3–35,682.",
    )
    AveOccup: float = Field(
        ...,
        ge=0.0,
        description="Average number of household members. Range: ~0.7–1,244.",
    )
    Latitude: float = Field(
        ...,
        ge=32.0,
        le=42.0,
        description="Block group latitude (California: 32.5–42.0).",
    )
    Longitude: float = Field(
        ...,
        le=-114.0,
        ge=-125.0,
        description="Block group longitude (California: −124.4 to −114.3).",
    )

    @field_validator("AveRooms")
    @classmethod
    def rooms_must_exceed_bedrooms(cls, v: float) -> float:

        if v < 1.0:
            raise ValueError("AveRooms must be >= 1.0 (a dwelling must have at least one room).")

        return v

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "MedInc": 3.5,
                "HouseAge": 28.0,
                "AveRooms": 5.2,
                "AveBedrms": 1.1,
                "Population": 1200.0,
                "AveOccup": 3.0,
                "Latitude": 37.88,
                "Longitude": -122.23,
            }
        },
    )


class PredictionResponse(BaseModel):
    """Response for a single prediction request."""

    predicted_value: float = Field(
        ...,
        description="Predicted median house value in units of $100,000. "
        "Multiply by 100,000 for USD. E.g., 2.5 → $250,000.",
    )

    predicted_value_usd: float = Field(
        ...,
        description="Predicted median house value in USD.",
    )

    model_name: str = Field(..., description="Name of the model that generated this prediction")
    model_version: str = Field(..., description="Version of the deployed model")

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "predicted_value": 2.478,
                "predicted_value_usd": 247800.0,
                "model_name": "california_housing_regressor",
                "model_version": "1.0.0",
            }
        },
    )


class BatchPredictionRequest(BaseModel):
    """Request body for batch prediction (up to 1000 samples)."""

    instances: list[HousingFeatures] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of housing feature objects.Maximum 1000 per request",
    )


class BatchPredictionResponse(BaseModel):
    """Response for a batch prediction request."""

    predictions: list[float] = Field(
        ...,
        description="List of predicted median house values (units $100.000),one per input response",
    )

    predictions_usd: list[float] = Field(..., description="Prediction in USD")

    n_samples: int = Field(..., description="Number of predictions returned")
    model_name: str = Field(..., description="Name of the model used.")
    model_version: str = Field(..., description="Version of the model deployed")

    model_config = ConfigDict(protected_namespaces=())


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="'ok' if the service is healthy")
    model_loaded: bool = Field(..., description="Whether the model is loaded and ready.")
    model_name: str = Field(..., description="Model Identifier")
    model_version: str = Field(..., description="Model Version")

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "status": "ok",
                "model_loaded": True,
                "model_name": "california_housing_regressor",
                "model_version": "1.0.0",
            }
        },
    )


class ErrorResponse(BaseModel):
    """Standard error response body."""

    error: str = Field(..., description="Short error type identifier")
    detail: str = Field(..., description="Human readable error description")
    status_code: int = Field(..., description="HTTP status code")
