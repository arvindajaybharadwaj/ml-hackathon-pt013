from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from data_pipeline import load_inputs, build_candidates, CandidateBatch

ROOT_DIR = Path(__file__).resolve().parents[2] / "Student folder"


def test_build_candidates_basic():
    inputs = load_inputs(ROOT_DIR, "train", "00000001")
    batch = build_candidates(inputs, config={"search_radius_m": 50.0, "max_candidates_per_obs": 15})
    
    assert isinstance(batch, CandidateBatch)
    assert len(batch.observations) == len(inputs.observations)
    assert len(batch.candidates) > 0
    
    # Check required columns
    expected_cols = [
        "trip_id", "obs_id", "candidate_id", "arc_id",
        "snapped_lon", "snapped_lat", "projection_fraction"
    ]
    for c in expected_cols:
        assert c in batch.candidates.columns
        
    # Check projection fractions
    assert (batch.candidates["projection_fraction"] >= 0.0).all()
    assert (batch.candidates["projection_fraction"] <= 1.0).all()
    
    # Check max candidates limit per obs
    counts_per_obs = batch.candidates.groupby("obs_id")["candidate_id"].count()
    assert (counts_per_obs <= 15).all()


def test_build_candidates_preserves_zero_candidates():
    inputs = load_inputs(ROOT_DIR, "train", "00000001")
    # Subsample to tiny search radius where some points may have zero candidates
    tiny_cfg = {"search_radius_m": 0.001, "max_candidates_per_obs": 5}
    batch = build_candidates(inputs, config=tiny_cfg)
    
    # All observations must still remain in batch.observations
    assert len(batch.observations) == len(inputs.observations)


def test_build_candidates_determinism():
    inputs = load_inputs(ROOT_DIR, "student_test", "00000000")
    b1 = build_candidates(inputs, config={"search_radius_m": 40.0, "max_candidates_per_obs": 10})
    b2 = build_candidates(inputs, config={"search_radius_m": 40.0, "max_candidates_per_obs": 10})
    
    assert len(b1.candidates) == len(b2.candidates)
    pd.testing.assert_frame_equal(b1.candidates, b2.candidates)
