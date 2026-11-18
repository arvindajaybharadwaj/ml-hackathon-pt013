from pathlib import Path
import pytest
from data_pipeline import load_inputs, build_candidates

ROOT_DIR = Path(__file__).resolve().parents[2] / "Student folder"


def test_no_leakage_in_candidate_batch():
    """Verify that candidate generation runs strictly reference-free."""
    inputs = load_inputs(ROOT_DIR, "train", "00000001")
    batch = build_candidates(inputs)
    
    # Check that candidates do not contain ground truth labels
    forbidden_cols = ["reference_arc_id", "is_ground_truth", "label", "route", "matched"]
    for col in forbidden_cols:
        assert col not in batch.candidates.columns, f"Leaked column {col} in CandidateBatch!"
    
    # Candidate features must not contain ground truth in observations either
    for col in forbidden_cols:
        assert col not in batch.observations.columns, f"Leaked column {col} in observations!"
