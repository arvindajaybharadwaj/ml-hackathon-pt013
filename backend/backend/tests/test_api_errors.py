from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app, build_services


def _settings(tmp_path):
    partition = tmp_path / "student_test" / "00000052"
    partition.mkdir(parents=True)
    (partition / "00000052.track").write_text("0.00010 0.00000 1\n0.00020 0.00000 2\n")
    (partition / "00000052.nodes").write_text("0.00000 0.00000\n0.00030 0.00000\n")
    (partition / "00000052.arcs").write_text("0 1\n")
    (partition / "00000052.route").write_text("0\n")
    return Settings(
        data_root=tmp_path, partition="student_test", candidate_dir=None,
        model_path=None, model_features=(), model_mode="distance_fallback",
        positive_class="1", allow_dev_fallback=True, candidate_radius_m=100,
        max_candidates=3, emission_sigma_m=25, transition_sigma_base_m=30,
        transition_sigma_fraction=.5, missing_state_penalty=6,
        unreachable_penalty=12, cors_origins=(),
    )


def test_invalid_and_unknown_trip_responses(tmp_path):
    with TestClient(create_app(_settings(tmp_path))) as client:
        assert client.get("/api/trips/not-an-id").status_code == 400
        assert client.get("/api/trips/99999999").status_code == 404
        assert client.get("/health").json()["prediction_mode"] == "development_fallback"


def test_production_rejects_development_fallback(tmp_path):
    settings = replace(_settings(tmp_path), environment="production")
    try:
        build_services(settings)
    except RuntimeError as exc:
        assert "PATHSNAP_CANDIDATE_DIR" in str(exc) or "development-only" in str(exc)
    else:
        raise AssertionError("Production must reject development fallback configuration")
