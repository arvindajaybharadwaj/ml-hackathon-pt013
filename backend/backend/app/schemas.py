from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    partition: str
    model_loaded: bool
    predictor: str
    predictor_mode: str
    candidate_provider: str
    data_root_exists: bool


class TripSummary(BaseModel):
    trip_id: str
    partition: str
    gps_observations: int
    road_nodes: int
    road_arcs: int
    bounds: dict[str, float]


class PredictionResponse(BaseModel):
    trip_id: str
    partition: str
    prediction_mode: str
    predictor: str
    gps_observations: int
    matched_observations: int
    missing_observations: int
    predicted_arc_ids: list[int | None]
    classifier_only_arc_ids: list[int | None]
    route_continuity: float
    classifier_only_route_continuity: float
    route_coordinates: list[list[float]]
    matches: list[dict]
    viterbi_log_probability: float
    processing_time_ms: float
    coordinate_order: str = "[longitude, latitude]"


class EvaluationResponse(BaseModel):
    trip_id: str
    prediction_mode: str
    predicted_sequence_length: int
    reference_sequence_length: int
    sequence_edit_similarity: float
    classifier_only_sequence_edit_similarity: float
    hmm_route_continuity: float
    classifier_only_route_continuity: float
    classifier_only_arc_set_f1: float
    lcs_recall_against_reference: float
    arc_set_precision: float
    arc_set_recall: float
    arc_set_f1: float
    note: str


class ErrorResponse(BaseModel):
    detail: str = Field(...)
