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
    dataset_available: bool = False
    hmm_enabled: bool = True
    prediction_mode: str


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
    candidate_provider: str
    gps_observations: int
    candidate_rows: int
    matched_observations: int
    missing_observations: int
    predicted_arc_ids: list[int | None]
    classifier_only_arc_ids: list[int | None]
    route_continuity: float
    classifier_only_route_continuity: float
    route_coordinates: list[list[float]]
    route_segments: list[list[list[float]]] = Field(default_factory=list)
    predicted_route_arc_ids: list[int | None] = Field(default_factory=list)
    classifier_route_arc_ids: list[int | None] = Field(default_factory=list)
    matches: list[dict]
    viterbi_log_probability: float
    decoder_diagnostics: dict
    timing_breakdown_ms: dict[str, float]
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
    timing_breakdown_ms: dict[str, float]
    decoder_diagnostics: dict
    note: str
    alignment_coverage: float | None = None
    evaluated_observations: int = 0
    classifier_correct_observations: int = 0
    hmm_correct_observations: int = 0
    classifier_top1_accuracy: float | None = None
    hmm_top1_accuracy: float | None = None
    classifier_observation_arc_macro_f1: float | None = None
    hmm_observation_arc_macro_f1: float | None = None


class ErrorResponse(BaseModel):
    detail: str = Field(...)
