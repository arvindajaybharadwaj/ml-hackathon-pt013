"""Example usage demonstrating Person 1 handoff interface for Persons 2 and 3.

Demonstrates:
1. load_inputs(root, partition, trip_id) -> TripInputs
2. build_candidates(inputs, config) -> CandidateBatch
3. load_labels(root, partition, trip_id) -> LabelTable
"""

import sys
from pathlib import Path
import json

# Ensure data_pipeline is importable
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from data_pipeline import load_inputs, build_candidates, load_labels, FEATURE_NAMES


def main():
    root = Path("F:/ML_QUIZ/Student folder/Student folder")
    partition = "train"
    trip_id = "00000001"
    
    print(f"=== PathSnap Data Pipeline Demonstration ===")
    print(f"Loading trip_id='{trip_id}' from partition='{partition}'...")
    
    # 1. Load inputs (no labels, reference-free)
    inputs = load_inputs(root, partition, trip_id)
    print(f"\n[1] TripInputs Loaded:")
    print(f"  - Trip ID: {inputs.trip_id}")
    print(f"  - Partition: {inputs.partition}")
    print(f"  - Observations: {len(inputs.observations):,} rows")
    print(f"  - Nodes: {len(inputs.nodes):,} rows")
    print(f"  - Arcs: {len(inputs.arcs):,} rows")
    print("  First 3 observations:")
    print(inputs.observations.head(3))
    
    # 2. Build candidates
    config = {
        "search_radius_m": 50.0,
        "max_candidates_per_obs": 15,
        "schema_version": "1.0.0"
    }
    print(f"\n[2] Building Candidates (search_radius={config['search_radius_m']}m, max_candidates={config['max_candidates_per_obs']})...")
    candidate_batch = build_candidates(inputs, config)
    print(f"  - Total Candidates Generated: {len(candidate_batch.candidates):,}")
    print(f"  - Observations preserved: {len(candidate_batch.observations):,}")
    print(f"  - Mean Candidates per Observation: {len(candidate_batch.candidates) / len(candidate_batch.observations):.2f}")
    print(f"  - Schema Version: {candidate_batch.schema_version}")
    print(f"  - Feature count: {len(FEATURE_NAMES)}")
    print("  First 3 candidates with spatial & kinematic features:")
    preview_cols = ["obs_id", "candidate_id", "arc_id", "distance_to_segment_m", "projection_fraction", "heading_diff_deg", "gps_speed_mps"]
    print(candidate_batch.candidates[preview_cols].head(3))
    
    # 3. Load labels (separate from inference / candidates)
    print(f"\n[3] Loading Separate Ground-Truth Labels...")
    label_table = load_labels(root, partition, trip_id)
    print(f"  - Total Observations Labeled: {len(label_table.labels):,}")
    print(f"  - Alignment Valid Coverage: {label_table.coverage * 100:.2f}%")
    print("  Label Table Reason Breakdown:")
    print(label_table.labels["alignment_reason"].value_counts().to_string())
    print("\n  First 3 label rows:")
    print(label_table.labels.head(3))
    
    print("\n=== Person 1 Pipeline Demonstration Completed Successfully! ===")


if __name__ == "__main__":
    main()
