"""Real artifact-to-API tests. Download the Modal run before running these."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from backend.app.config import Settings
from backend.app.data import ReferenceRouteRepository
from backend.app.main import create_app
from backend.app.predictor import BundleCandidatePredictor
from ml.predictor import load_model

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / 'artifacts/model'


@pytest.fixture(scope='module')
def client():
    if not MODEL.is_dir():
        pytest.skip('Download the completed Modal model as described in README.md')
    settings = replace(Settings.from_env(), data_root=ROOT / 'Student folder',
                       partition='student_test', candidate_dir=ROOT / 'prepared_data/student_test',
                       model_path=MODEL, model_mode='bundle', environment='production')
    with TestClient(create_app(settings)) as session:
        yield session


def test_real_bundle_predictions_evaluation_map_and_label_isolation(client, monkeypatch):
    health = client.get('/health').json()
    assert health['status'] == 'healthy'
    assert health['predictor'] == 'hist_gradient_boosting'
    assert health['model_loaded'] is True
    assert len(client.get('/api/trips').json()['trip_ids']) == 10
    assert client.get('/api/trips/not-a-trip').status_code == 400
    original = ReferenceRouteRepository.load_reference_route
    def forbidden(*args):
        raise AssertionError('Inference or maps must not load reference routes')
    monkeypatch.setattr(ReferenceRouteRepository, 'load_reference_route', forbidden)
    response = client.post('/api/predict/00000052')
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['prediction_mode'] == 'trained_model'
    assert result['gps_observations'] == len(result['predicted_arc_ids']) == 262
    assert result['candidate_rows'] > 0
    assert result['missing_observations'] == 0
    assert 'reference_arc_id' not in response.text
    map_response = client.get('/api/maps/00000052')
    assert map_response.status_code == 200
    assert 'HMM matched roads' in map_response.text
    assert 'GPS observations' in map_response.text
    assert 'reference_arc_id' not in map_response.text
    monkeypatch.setattr(ReferenceRouteRepository, 'load_reference_route', original)
    evaluation = client.get('/api/evaluate/00000052')
    assert evaluation.status_code == 200
    assert 0 <= evaluation.json()['arc_set_f1'] <= 1
    # Prediction and map reuse the decoded result rather than repeating expensive HMM work.
    assert client.app.state.prediction.predict.cache_info().hits >= 2


def test_backend_bundle_matches_ml_scores_and_checks_schema(client):
    candidates = pd.read_parquet(ROOT / 'prepared_data/student_test/candidates.parquet',
                                 filters=[('trip_id', '=', '00000052')]).head(100)
    bundle = load_model(MODEL)
    version = bundle.schema['schema_version']
    expected = bundle.score_candidates({'schema_version': version, 'candidates': candidates})
    candidates.attrs['schema_version'] = version
    adapter = BundleCandidatePredictor(MODEL)
    np.testing.assert_allclose(adapter.score_candidates(candidates), expected.p_match)
    candidates.attrs['schema_version'] = 'wrong'
    with pytest.raises(ValueError, match='version'):
        adapter.score_candidates(candidates)
