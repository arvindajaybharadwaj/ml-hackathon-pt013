import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from data_pipeline import load_inputs, build_candidates, FEATURE_NAMES

ROOT_DIR = Path(__file__).resolve().parents[2] / "Student folder"
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "feature_schema.json"


def test_feature_schema_matches_code():
    with open(SCHEMA_PATH, "r") as f:
        schema = json.load(f)
    assert schema["feature_order"] == FEATURE_NAMES


def test_features_presence_and_no_nan():
    inputs = load_inputs(ROOT_DIR, "train", "00000001")
    batch = build_candidates(inputs)
    
    # Check that all features are present
    for feat in FEATURE_NAMES:
        assert feat in batch.candidates.columns
        # Check no NaN values
        assert not batch.candidates[feat].isna().any(), f"NaN found in feature {feat}"
        
    # Check physical ranges
    assert (batch.candidates["distance_to_segment_m"] >= 0.0).all()
    assert (batch.candidates["distance_to_segment_m"] <= 50.0).all()
    assert (batch.candidates["heading_diff_deg"] >= 0.0).all()
    assert (batch.candidates["heading_diff_deg"] <= 180.0).all()
    assert (batch.candidates["heading_alignment_cos"] >= -1.0).all()
    assert (batch.candidates["heading_alignment_cos"] <= 1.0).all()
    assert set(batch.candidates["reliability_flag"].unique()).issubset({0, 1})
    assert set(batch.candidates["is_zero_speed"].unique()).issubset({0, 1})
