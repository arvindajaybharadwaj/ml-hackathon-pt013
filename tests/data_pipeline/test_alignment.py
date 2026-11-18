from pathlib import Path
import pytest
import pandas as pd
from data_pipeline import load_inputs, load_labels, load_raw_route, align_route_to_observations, LabelTable

ROOT_DIR = Path(__file__).resolve().parents[2] / "Student folder"


def test_alignment_train_trip():
    labels_table = load_labels(ROOT_DIR, "train", "00000001")
    assert isinstance(labels_table, LabelTable)
    assert labels_table.trip_id == "00000001"
    assert labels_table.partition == "train"
    assert len(labels_table.labels) == 1315
    assert labels_table.coverage > 0.95
    
    # Check columns
    expected_cols = ["trip_id", "obs_id", "reference_arc_id", "alignment_valid", "alignment_reason"]
    assert list(labels_table.labels.columns) == expected_cols
    
    # Check validity consistency
    valid_rows = labels_table.labels[labels_table.labels["alignment_valid"] == True]
    assert not valid_rows["reference_arc_id"].isna().any()


def test_alignment_with_missing_route():
    inputs = load_inputs(ROOT_DIR, "train", "00000001")
    empty_route = pd.DataFrame(columns=["arc_id"])
    res = align_route_to_observations(inputs, empty_route)
    assert len(res.labels) == len(inputs.observations)
    assert (res.labels["alignment_valid"] == False).all()
    assert res.labels["reference_arc_id"].isna().all()
