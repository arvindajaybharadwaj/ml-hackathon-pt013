from pathlib import Path
import pytest
import pandas as pd
from data_pipeline import load_inputs, load_labels, load_raw_route, TripInputs, LabelTable

ROOT_DIR = Path(__file__).resolve().parents[2] / "Student folder"


def test_load_inputs_train():
    inputs = load_inputs(ROOT_DIR, "train", "00000001")
    assert isinstance(inputs, TripInputs)
    assert inputs.trip_id == "00000001"
    assert inputs.partition == "train"
    
    # Check observations
    assert list(inputs.observations.columns) == ["obs_id", "lon", "lat", "timestamp_s"]
    assert len(inputs.observations) > 0
    assert inputs.observations["obs_id"].iloc[0] == 0
    assert inputs.observations["obs_id"].iloc[-1] == len(inputs.observations) - 1
    
    # Check nodes
    assert list(inputs.nodes.columns) == ["node_id", "lon", "lat"]
    assert len(inputs.nodes) > 0
    assert inputs.nodes["node_id"].iloc[0] == 0
    
    # Check arcs
    assert list(inputs.arcs.columns) == ["arc_id", "from_node_id", "to_node_id"]
    assert len(inputs.arcs) > 0
    assert inputs.arcs["arc_id"].iloc[0] == 0


def test_load_inputs_student_test():
    inputs = load_inputs(ROOT_DIR, "student_test", "00000000")
    assert isinstance(inputs, TripInputs)
    assert inputs.trip_id == "00000000"
    assert inputs.partition == "student_test"
    assert len(inputs.observations) == 2503


def test_load_raw_route():
    route = load_raw_route(ROOT_DIR, "train", "00000001")
    assert route is not None
    assert "arc_id" in route.columns
    assert len(route) == 525


def test_no_leakage_in_trip_inputs():
    inputs = load_inputs(ROOT_DIR, "train", "00000001")
    forbidden = ["route", "arc_id", "reference_arc_id", "label"]
    for col in forbidden:
        assert col not in inputs.observations.columns
