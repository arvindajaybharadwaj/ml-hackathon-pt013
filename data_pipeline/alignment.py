"""Order-aware alignment of GPS observations to ground-truth reference routes.

Implements Task 5:
- Route rows do not correspond one-to-one with GPS rows.
- Uses an order-aware forward window along the chronological route sequence.
- Naturally resolves repeated road segments (distinguishes 1st pass from 2nd pass).
- Explicitly identifies unmatchable / uncertain observations (e.g. trajectory beyond route).
- Produces a LabelTable with nullable reference_arc_id and alignment validity diagnostics.
"""

from typing import Optional
import numpy as np
import pandas as pd

from .contracts import TripInputs, LabelTable


def align_route_to_observations(inputs: TripInputs,
                                raw_route: pd.DataFrame,
                                max_dist_m: float = 35.0,
                                search_window: int = 15) -> LabelTable:
    """Align GPS observations to the chronological sequence of reference route arcs.
    
    Args:
        inputs: TripInputs containing observations, nodes, arcs.
        raw_route: DataFrame containing ['route_step', 'arc_id'] or single column of arc_ids.
        max_dist_m: Maximum perpendicular distance in metres for an observation to be considered
                    validly matched to a route segment (default: 35.0m).
        search_window: Number of forward route segments to search around the current route pointer.
        
    Returns:
        LabelTable containing:
            ['trip_id', 'obs_id', 'reference_arc_id', 'alignment_valid', 'alignment_reason']
    """
    track_df = inputs.observations
    nodes_df = inputs.nodes
    arcs_df = inputs.arcs
    
    # Extract route arcs array
    if "arc_id" in raw_route.columns:
        route_arcs = raw_route["arc_id"].values.astype(int)
    else:
        route_arcs = raw_route.iloc[:, 0].values.astype(int)
        
    M = len(route_arcs)
    T = len(track_df)
    
    if M == 0 or T == 0:
        empty_labels = pd.DataFrame({
            "trip_id": [inputs.trip_id] * T,
            "obs_id": np.arange(T, dtype=int),
            "reference_arc_id": pd.Series([pd.NA] * T, dtype="Int64"),
            "alignment_valid": [False] * T,
            "alignment_reason": ["empty_route_or_observations"] * T
        })
        return LabelTable(labels=empty_labels, trip_id=inputs.trip_id, partition=inputs.partition, coverage=0.0)

    # Local metric projection
    mean_lat = float(track_df["lat"].mean())
    mean_lon = float(track_df["lon"].mean())
    kx = 111320.0 * np.cos(np.radians(mean_lat))
    ky = 111320.0
    
    node_x = (nodes_df["lon"].values - mean_lon) * kx
    node_y = (nodes_df["lat"].values - mean_lat) * ky
    track_x = (track_df["lon"].values - mean_lon) * kx
    track_y = (track_df["lat"].values - mean_lat) * ky
    
    # Precompute line segment endpoints for all route arcs
    u = arcs_df.iloc[route_arcs]["from_node_id"].values
    v = arcs_df.iloc[route_arcs]["to_node_id"].values
    ax = node_x[u]
    ay = node_y[u]
    bx = node_x[v]
    by = node_y[v]
    dx = bx - ax
    dy = by - ay
    seg_len_sq = np.maximum(dx**2 + dy**2, 1e-6)
    
    # Initialize route pointer: find best start arc within first 15 route arcs
    init_w = min(15, M)
    p0_x, p0_y = track_x[0], track_y[0]
    t0 = np.clip(((p0_x - ax[:init_w])*dx[:init_w] + (p0_y - ay[:init_w])*dy[:init_w]) / seg_len_sq[:init_w], 0.0, 1.0)
    d0 = np.sqrt((p0_x - (ax[:init_w] + t0*dx[:init_w]))**2 + (p0_y - (ay[:init_w] + t0*dy[:init_w]))**2)
    r_idx = int(np.argmin(d0))
    
    aligned_arcs = []
    alignment_valid = []
    alignment_reasons = []
    
    time_arr = track_df["timestamp_s"].values
    
    for t in range(T):
        px = track_x[t]
        py = track_y[t]
        
        # Check for large time gap from previous observation
        dt = time_arr[t] - time_arr[t-1] if t > 0 else 1.0
        # If there was a large time gap (>60s), expand search window forward
        dyn_window = search_window if dt < 10.0 else min(M - r_idx, int(search_window + dt * 0.5))
        
        # Local search window along chronological route
        w_start = max(0, r_idx - 1) # allow small 1-step backward jitter for GPS noise
        w_end = min(M, r_idx + dyn_window)
        
        t_proj = np.clip(
            ((px - ax[w_start:w_end])*dx[w_start:w_end] + (py - ay[w_start:w_end])*dy[w_start:w_end]) / seg_len_sq[w_start:w_end],
            0.0, 1.0
        )
        d_candidates = np.sqrt(
            (px - (ax[w_start:w_end] + t_proj*dx[w_start:w_end]))**2 +
            (py - (ay[w_start:w_end] + t_proj*dy[w_start:w_end]))**2
        )
        
        best_local = int(np.argmin(d_candidates))
        best_d = float(d_candidates[best_local])
        best_r = w_start + best_local
        
        if best_d <= max_dist_m:
            r_idx = best_r  # advance route pointer
            matched_arc = int(route_arcs[best_r])
            aligned_arcs.append(matched_arc)
            alignment_valid.append(True)
            alignment_reasons.append("valid_aligned")
        else:
            # Trajectory is far from expected route segment
            if r_idx >= M - 2:
                aligned_arcs.append(pd.NA)
                alignment_valid.append(False)
                alignment_reasons.append("route_extent_exceeded")
            elif dt > 60.0:
                aligned_arcs.append(pd.NA)
                alignment_valid.append(False)
                alignment_reasons.append("temporal_gap_break")
            else:
                aligned_arcs.append(pd.NA)
                alignment_valid.append(False)
                alignment_reasons.append("distance_exceeds_threshold")
                
    labels_df = pd.DataFrame({
        "trip_id": [str(inputs.trip_id)] * T,
        "obs_id": np.arange(T, dtype=int),
        "reference_arc_id": pd.Series(aligned_arcs, dtype="Int64"),
        "alignment_valid": alignment_valid,
        "alignment_reason": alignment_reasons
    })
    
    coverage = float(np.mean(alignment_valid))
    return LabelTable(
        labels=labels_df,
        trip_id=str(inputs.trip_id),
        partition=str(inputs.partition),
        coverage=coverage
    )
