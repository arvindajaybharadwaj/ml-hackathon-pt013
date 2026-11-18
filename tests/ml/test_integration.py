"""Real handoff integration and regression tests; no dummy training inputs."""
from pathlib import Path
import copy
import json
import numpy as np
import pandas as pd
import pytest
from ml.data import Dataset, load_dataset, KEY
from ml.models import MODEL_NAMES
from ml.train import fit, measure
from ml.predictor import load_model

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def real():
    full = load_dataset(ROOT / 'prepared_data')
    assert full.observations.groupby('partition').trip_id.nunique().to_dict() == {'student_test': 10, 'train': 80}
    assert len(full.schema['feature_names']) == 13
    obs = full.observations.groupby('trip_id', sort=True).head(30)
    ids = sorted(obs.loc[obs.partition.eq('train'), 'trip_id'].unique())[:3]
    ids += sorted(obs.loc[obs.partition.eq('student_test'), 'trip_id'].unique())[:1]
    obs = obs[obs.trip_id.isin(ids)]
    return Dataset(obs, full.candidates.merge(obs[KEY], on=KEY),
                   full.labels.merge(obs[KEY], on=KEY), full.schema).validate()


@pytest.mark.parametrize('name', MODEL_NAMES + ['torch_mlp'])
def test_real_training_reload_and_backend_contract(real, name, tmp_path):
    predictor, _ = fit(real.partition('train'), name, 13, options={'device': 'cpu', 'epochs': 2})
    test = real.partition('student_test')
    metrics, before, _ = measure(predictor, test)
    assert 0 <= metrics['top1_accuracy'] <= 1
    np.testing.assert_allclose(before.groupby(KEY).emission_prob.sum(), 1)
    predictor.save(tmp_path)
    after = load_model(tmp_path).score_candidates(test)
    np.testing.assert_allclose(before.p_match, after.p_match)
    # CandidateBatch is the real upstream inference interface.
    from data_pipeline.contracts import CandidateBatch
    batch = CandidateBatch(test.observations, test.candidates, real.schema['schema_version'])
    np.testing.assert_allclose(before.p_match, predictor.score_candidates(batch).p_match)
    changed = copy.deepcopy(test)
    changed.labels['reference_arc_id'] = 999999
    np.testing.assert_allclose(before.p_match, predictor.score_candidates(changed).p_match)


def test_schema_mismatch_and_identifier_features_rejected(real):
    bad = copy.deepcopy(real)
    bad.schema['feature_names'] = ['arc_id']
    with pytest.raises(ValueError):
        bad.validate()
    predictor, _ = fit(real.partition('train'), 'gaussian_nb', 13)
    with pytest.raises(ValueError, match='version'):
        predictor.score_candidates({'schema_version': 'wrong', 'candidates': real.candidates})


def test_nullable_reference_missing_predictions_count_wrong():
    obs = pd.DataFrame({'trip_id': ['a', 'a'], 'obs_id': [0, 1], 'partition': ['train'] * 2})
    candidates = pd.DataFrame({'trip_id': ['a'], 'obs_id': [0], 'candidate_id': [0], 'arc_id': [5], 'distance': [1.]})
    labels = obs[KEY].assign(reference_arc_id=pd.Series([5, 6], dtype='Int64'), alignment_valid=True)
    ds = Dataset(obs, candidates, labels, {'schema_version': 'v1', 'feature_names': ['distance']})
    scores = candidates.drop(columns='distance').assign(p_match=.9, emission_prob=1.)
    from ml.metrics import evaluate
    metrics, _ = evaluate(ds, scores)
    assert metrics['top1_accuracy'] == .5
    assert metrics['candidate_miss_rate'] == .5
    assert metrics['zero_candidate_observations'] == 1
    with pytest.raises(ValueError, match='exactly'):
        evaluate(ds, scores.iloc[:0])


def test_trip_grouping_and_preprocessing_do_not_leak(real):
    from sklearn.model_selection import GroupKFold
    train = real.partition('train')
    for a, b in GroupKFold(3).split(train.observations, groups=train.observations.trip_id):
        assert set(train.observations.iloc[a].trip_id).isdisjoint(train.observations.iloc[b].trip_id)
    predictor, _ = fit(train, 'logistic_regression', 13)
    stats = predictor.pipeline.named_steps['imputer'].statistics_.copy()
    test = copy.deepcopy(real.partition('student_test'))
    test.candidates[real.schema['feature_names']] = 1e8
    predictor.score_candidates(test)
    np.testing.assert_array_equal(stats, predictor.pipeline.named_steps['imputer'].statistics_)
