"""Data pipeline data structures and handoff types for PathSnap.

Defines the exact interface contracts for Person 1 handoff to Persons 2 and 3:
- TripInputs
- CandidateBatch
- LabelTable
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional
import pandas as pd


@dataclass
class TripInputs:
    """Raw loaded trip inputs without any labels or reference routes.
    
    Attributes:
        observations: pd.DataFrame with columns ['obs_id', 'lon', 'lat', 'timestamp_s']
        nodes: pd.DataFrame with columns ['node_id', 'lon', 'lat']
        arcs: pd.DataFrame with columns ['arc_id', 'from_node_id', 'to_node_id']
        trip_id: Unique string identifier for the trip (e.g. '00000001')
        partition: Partition name ('train' or 'student_test')
    """
    observations: pd.DataFrame
    nodes: pd.DataFrame
    arcs: pd.DataFrame
    trip_id: str
    partition: str

    def __post_init__(self):
        # Validate that no labels exist in TripInputs
        for col in ['route', 'arc_id', 'reference_arc_id', 'label']:
            if col in self.observations.columns:
                raise ValueError(f"Data leakage detected: {col} column found in TripInputs.observations!")


@dataclass
class CandidateBatch:
    """Road segment candidates generated for an observation trajectory.
    
    Attributes:
        observations: pd.DataFrame with all original GPS observations (preserves zero-candidate rows)
        candidates: pd.DataFrame with candidate rows containing:
            ['trip_id', 'obs_id', 'candidate_id', 'arc_id', 'snapped_lon', 'snapped_lat',
             'projection_fraction', <feature columns>]
        schema_version: Semantic version of the feature schema
    """
    observations: pd.DataFrame
    candidates: pd.DataFrame
    schema_version: str = "1.0.0"

    def __post_init__(self):
        # Validate no ground truth labels leaked into candidates
        for col in ['reference_arc_id', 'is_ground_truth', 'label']:
            if col in self.candidates.columns:
                raise ValueError(f"Data leakage detected: {col} column found in CandidateBatch.candidates!")


@dataclass
class LabelTable:
    """Order-aware aligned reference route labels. Stored separately from candidate features.
    
    Attributes:
        labels: pd.DataFrame with columns:
            ['trip_id', 'obs_id', 'reference_arc_id', 'alignment_valid', 'alignment_reason']
        trip_id: Unique string identifier for the trip
        partition: Partition name
        coverage: Fraction of observations with valid alignment (0.0 to 1.0)
    """
    labels: pd.DataFrame
    trip_id: str
    partition: str
    coverage: float = 0.0
