from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app


def _write_trip(root: Path, partition: str, trip_id: str):
    folder = root / partition / trip_id
    folder.mkdir(parents=True)
    (folder / f"{trip_id}.track").write_text(
        "0.00010 0.00000 1\n0.00020 0.00000 2\n0.00030 0.00000 3\n"
    )
    (folder / f"{trip_id}.nodes").write_text(
        "0.00000 0.00000\n0.00020 0.00000\n0.00040 0.00000\n"
    )
    (folder / f"{trip_id}.arcs").write_text("0 1\n1 2\n")
    (folder / f"{trip_id}.route").write_text("0\n1\n")


def test_api_smoke(tmp_path):
    _write_trip(tmp_path, "student_test", "00000052")
    settings = Settings(
        data_root=tmp_path, partition="student_test", candidate_dir=None,
        model_path=None, model_features=(), model_mode="distance_fallback",
        positive_class="1", allow_dev_fallback=True,
        candidate_radius_m=100.0, max_candidates=3, emission_sigma_m=25.0,
        transition_sigma_base_m=30.0, transition_sigma_fraction=0.5,
        missing_state_penalty=6.0, unreachable_penalty=12.0,
        cors_origins=("http://localhost:5173",),
    )
    app = create_app(settings)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/api/trips").json()["trip_ids"] == ["00000052"]
        assert client.get("/api/trips/00000052").status_code == 200
        prediction = client.post("/api/predict/00000052")
        assert prediction.status_code == 200, prediction.text
        body = prediction.json()
        assert len(body["predicted_arc_ids"]) == 3
        assert body["prediction_mode"] == "development_fallback"
        evaluation = client.get("/api/evaluate/00000052")
        assert evaluation.status_code == 200, evaluation.text
        assert "sequence_edit_similarity" in evaluation.json()
