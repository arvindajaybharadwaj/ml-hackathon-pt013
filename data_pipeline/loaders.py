"""Raw loaders for PathSnap datasets.

Implements Task 2 and the Person 1 handoff contract:
- load_inputs(root, partition, trip_id) -> TripInputs
- load_labels(root, partition, trip_id) -> LabelTable
- load_raw_route(root, partition, trip_id) -> pd.DataFrame
"""

from pathlib import Path
from typing import Union, Optional
import numpy as np
import pandas as pd

from .contracts import TripInputs, LabelTable


def _resolve_trip_dir(root: Union[str, Path], partition: str, trip_id: str) -> Path:
    """Resolve the trip directory path robustly, handling potential nested folders."""
    p_root = Path(root)
    # Check direct root / partition / trip_id
    candidate = p_root / partition / trip_id
    if candidate.exists() and candidate.is_dir():
        return candidate
    
    # Check nested root / 'Student folder' / partition / trip_id
    candidate_nested = p_root / "Student folder" / partition / trip_id
    if candidate_nested.exists() and candidate_nested.is_dir():
        return candidate_nested
    
    # Check parent
    candidate_parent = p_root.parent / partition / trip_id
    if candidate_parent.exists() and candidate_parent.is_dir():
        return candidate_parent

    raise FileNotFoundError(
        f"Trip directory for partition='{partition}', trip_id='{trip_id}' not found under root='{root}'."
    )


def load_inputs(root: Union[str, Path], partition: str, trip_id: str) -> TripInputs:
    """Load raw trip observations, nodes, and arcs into a TripInputs object.
    
    Guarantees:
    - Observations have ['obs_id', 'lon', 'lat', 'timestamp_s']
    - Nodes have ['node_id', 'lon', 'lat']
    - Arcs have ['arc_id', 'from_node_id', 'to_node_id']
    - Zero-based contiguous row indices preserved as IDs.
    - No reference route or ground-truth labels are included.
    
    Args:
        root: Path to the Student folder root directory.
        partition: 'train' or 'student_test'.
        trip_id: String trip identifier (e.g. '00000001').
        
    Returns:
        TripInputs object.
    """
    trip_dir = _resolve_trip_dir(root, partition, trip_id)
    
    # 1. Load Track
    track_path = trip_dir / f"{trip_id}.track"
    if not track_path.exists():
        raise FileNotFoundError(f"Missing track file: {track_path}")
    obs_df = pd.read_csv(track_path, sep=r"\s+", header=None, names=["lon", "lat", "timestamp_s"], dtype=float)
    obs_df.insert(0, "obs_id", np.arange(len(obs_df), dtype=int))
    
    # 2. Load Nodes
    nodes_path = trip_dir / f"{trip_id}.nodes"
    if not nodes_path.exists():
        raise FileNotFoundError(f"Missing nodes file: {nodes_path}")
    nodes_df = pd.read_csv(nodes_path, sep=r"\s+", header=None, names=["lon", "lat"], dtype=float)
    nodes_df.insert(0, "node_id", np.arange(len(nodes_df), dtype=int))
    
    # 3. Load Arcs
    arcs_path = trip_dir / f"{trip_id}.arcs"
    if not arcs_path.exists():
        raise FileNotFoundError(f"Missing arcs file: {arcs_path}")
    arcs_df = pd.read_csv(arcs_path, sep=r"\s+", header=None, names=["from_node_id", "to_node_id"], dtype=int)
    arcs_df.insert(0, "arc_id", np.arange(len(arcs_df), dtype=int))
    
    return TripInputs(
        observations=obs_df,
        nodes=nodes_df,
        arcs=arcs_df,
        trip_id=str(trip_id),
        partition=str(partition)
    )


def load_raw_route(root: Union[str, Path], partition: str, trip_id: str) -> Optional[pd.DataFrame]:
    """Load the raw chronological reference route arc sequence if available.
    
    Args:
        root: Path to the Student folder root directory.
        partition: 'train' or 'student_test'.
        trip_id: String trip identifier.
        
    Returns:
        DataFrame with columns ['route_step', 'arc_id'] or None if file absent.
    """
    try:
        trip_dir = _resolve_trip_dir(root, partition, trip_id)
    except FileNotFoundError:
        return None
        
    route_path = trip_dir / f"{trip_id}.route"
    if not route_path.exists():
        return None
        
    route_df = pd.read_csv(route_path, sep=r"\s+", header=None, names=["arc_id"], dtype=int)
    route_df.insert(0, "route_step", np.arange(len(route_df), dtype=int))
    return route_df


def load_labels(root: Union[str, Path], partition: str, trip_id: str,
                prepared_data_dir: Optional[Union[str, Path]] = None) -> LabelTable:
    """Load observation-level ground-truth labels for a trip.
    
    Checks for precomputed labels in prepared_data_dir first; if not provided or missing,
    computes an order-aware alignment using data_pipeline.alignment.align_route_to_observations.
    
    Args:
        root: Path to raw dataset root.
        partition: 'train' or 'student_test'.
        trip_id: String trip identifier.
        prepared_data_dir: Optional path to directory where prepared parquet tables are stored.
        
    Returns:
        LabelTable with ['trip_id', 'obs_id', 'reference_arc_id', 'alignment_valid', 'alignment_reason']
    """
    # 1. Try loading from prepared parquet if available
    if prepared_data_dir is not None:
        parquet_path = Path(prepared_data_dir) / partition / "labels.parquet"
        if parquet_path.exists():
            labels_all = pd.read_parquet(parquet_path)
            trip_labels = labels_all[labels_all["trip_id"] == str(trip_id)].copy()
            if not trip_labels.empty:
                coverage = float((trip_labels["alignment_valid"] == True).mean())
                return LabelTable(
                    labels=trip_labels.reset_index(drop=True),
                    trip_id=str(trip_id),
                    partition=str(partition),
                    coverage=coverage
                )
    
    # 2. Otherwise load raw files and align
    inputs = load_inputs(root, partition, trip_id)
    raw_route = load_raw_route(root, partition, trip_id)
    
    if raw_route is None or raw_route.empty:
        # Route absent (e.g. unlabelled test set)
        n_obs = len(inputs.observations)
        labels_df = pd.DataFrame({
            "trip_id": [str(trip_id)] * n_obs,
            "obs_id": np.arange(n_obs, dtype=int),
            "reference_arc_id": pd.Series([pd.NA] * n_obs, dtype="Int64"),
            "alignment_valid": [False] * n_obs,
            "alignment_reason": ["route_file_absent"] * n_obs
        })
        return LabelTable(
            labels=labels_df,
            trip_id=str(trip_id),
            partition=str(partition),
            coverage=0.0
        )
    
    from .alignment import align_route_to_observations
    return align_route_to_observations(inputs, raw_route)
