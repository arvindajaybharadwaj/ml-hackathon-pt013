"""Candidate road segment generation using spatial indexing and metric geometry.

Implements Task 3 and the Person 1 handoff contract:
- build_candidates(inputs, config) -> CandidateBatch
- Reference-free generation (does not read reference routes)
- Preserves zero-candidate observations in the observations table
- Computes snapped coordinates, projection fractions, and candidate features
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from shapely import STRtree, linestrings, points

from .contracts import TripInputs, CandidateBatch
from .features import (
    FEATURE_NAMES,
    calculate_trajectory_kinematics,
    calculate_bearing_deg,
    compute_candidate_features
)


DEFAULT_CONFIG: Dict[str, Any] = {
    "search_radius_m": 50.0,
    "max_candidates_per_obs": 15,
    "schema_version": "1.0.0"
}


def build_candidates(inputs: TripInputs, config: Optional[Dict[str, Any]] = None) -> CandidateBatch:
    """Generate candidate road segments for each GPS observation.
    
    Args:
        inputs: TripInputs containing observations, nodes, arcs.
        config: Optional configuration dictionary containing:
            - 'search_radius_m': float (default 50.0m)
            - 'max_candidates_per_obs': int (default 15)
            - 'schema_version': str (default '1.0.0')
            
    Returns:
        CandidateBatch containing original observations and candidates DataFrame.
    """
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
        
    search_radius_m = float(cfg["search_radius_m"])
    max_candidates = int(cfg["max_candidates_per_obs"])
    schema_version = str(cfg["schema_version"])
    
    obs_df = inputs.observations
    nodes_df = inputs.nodes
    arcs_df = inputs.arcs
    trip_id = str(inputs.trip_id)
    n_obs = len(obs_df)
    n_arcs = len(arcs_df)
    
    if n_obs == 0 or n_arcs == 0:
        empty_cands = pd.DataFrame(columns=[
            "trip_id", "obs_id", "candidate_id", "arc_id",
            "snapped_lon", "snapped_lat", "projection_fraction"
        ] + FEATURE_NAMES)
        return CandidateBatch(observations=obs_df, candidates=empty_cands, schema_version=schema_version)

    # 1. Local metric projection centered on trip observation mean
    mean_lat = float(obs_df["lat"].mean())
    mean_lon = float(obs_df["lon"].mean())
    kx = 111320.0 * np.cos(np.radians(mean_lat))
    ky = 111320.0
    
    node_x = (nodes_df["lon"].values - mean_lon) * kx
    node_y = (nodes_df["lat"].values - mean_lat) * ky
    track_x = (obs_df["lon"].values - mean_lon) * kx
    track_y = (obs_df["lat"].values - mean_lat) * ky
    time_s = obs_df["timestamp_s"].values
    
    # 2. Trajectory kinematics (observation level)
    speeds_mps, dt_s, step_dist_m, traj_bearings_deg, is_zero_speed = calculate_trajectory_kinematics(
        track_x, track_y, time_s
    )
    
    # 3. Build spatial index over directed road segments
    from_u = arcs_df["from_node_id"].values
    to_v = arcs_df["to_node_id"].values
    
    ax = node_x[from_u]
    ay = node_y[from_u]
    bx = node_x[to_v]
    by = node_y[to_v]
    
    dx = bx - ax
    dy = by - ay
    seg_len_sq = dx**2 + dy**2
    seg_len_m = np.sqrt(seg_len_sq).astype(np.float32)
    seg_bearings_deg = calculate_bearing_deg(ax, ay, bx, by).astype(np.float32)
    
    # Construct Shapely linestrings for STRtree
    coords = np.column_stack([ax, ay, bx, by]).reshape(-1, 2, 2)
    lines = linestrings(coords)
    tree = STRtree(lines)
    
    # 4. Query spatial index for all points within search_radius_m
    pts = points(np.column_stack([track_x, track_y]))
    pt_indices, arc_indices = tree.query(pts, predicate="dwithin", distance=search_radius_m)
    
    if len(pt_indices) == 0:
        empty_cands = pd.DataFrame(columns=[
            "trip_id", "obs_id", "candidate_id", "arc_id",
            "snapped_lon", "snapped_lat", "projection_fraction"
        ] + FEATURE_NAMES)
        return CandidateBatch(observations=obs_df, candidates=empty_cands, schema_version=schema_version)
        
    # 5. Compute exact metric point-to-segment geometry
    px = track_x[pt_indices]
    py = track_y[pt_indices]
    seg_ax = ax[arc_indices]
    seg_ay = ay[arc_indices]
    seg_dx = dx[arc_indices]
    seg_dy = dy[arc_indices]
    seg_len_sq_sub = np.maximum(seg_len_sq[arc_indices], 1e-6)
    
    ap_x = px - seg_ax
    ap_y = py - seg_ay
    
    proj_fraction = np.clip((ap_x * seg_dx + ap_y * seg_dy) / seg_len_sq_sub, 0.0, 1.0).astype(np.float32)
    
    snap_x = seg_ax + proj_fraction * seg_dx
    snap_y = seg_ay + proj_fraction * seg_dy
    
    dist_m = np.sqrt((px - snap_x)**2 + (py - snap_y)**2).astype(np.float32)
    
    # Inverse project snapped coordinates back to WGS84
    snapped_lon = (mean_lon + snap_x / kx).astype(float)
    snapped_lat = (mean_lat + snap_y / ky).astype(float)
    
    # 6. Filter by max_candidates per observation (keeping closest candidates)
    # We build an initial DataFrame to rank and prune
    df_raw = pd.DataFrame({
        "obs_id": pt_indices,
        "arc_id": arc_indices,
        "dist_m": dist_m,
        "proj_fraction": proj_fraction,
        "snapped_lon": snapped_lon,
        "snapped_lat": snapped_lat,
        "seg_len_m": seg_len_m[arc_indices],
        "seg_bearing": seg_bearings_deg[arc_indices]
    })
    
    # Sort by (obs_id, dist_m)
    df_raw = df_raw.sort_values(by=["obs_id", "dist_m"])
    
    # Deduplicate in case multiple identical arcs exist
    df_raw = df_raw.drop_duplicates(subset=["obs_id", "arc_id"])
    
    # Prune to max_candidates per obs_id
    df_raw["cand_rank"] = df_raw.groupby("obs_id").cumcount()
    df_pruned = df_raw[df_raw["cand_rank"] < max_candidates].copy()
    
    # Restore observation-level kinematics to candidate rows
    cand_obs_ids = df_pruned["obs_id"].values
    cand_speed = speeds_mps[cand_obs_ids]
    cand_dt = dt_s[cand_obs_ids]
    cand_dist = step_dist_m[cand_obs_ids]
    cand_bearing = traj_bearings_deg[cand_obs_ids]
    cand_zero_spd = is_zero_speed[cand_obs_ids]
    
    # 7. Calculate all features
    features_df = compute_candidate_features(
        obs_id_arr=cand_obs_ids,
        dist_m_arr=df_pruned["dist_m"].values,
        proj_frac_arr=df_pruned["proj_fraction"].values,
        seg_len_arr=df_pruned["seg_len_m"].values,
        seg_bearing_arr=df_pruned["seg_bearing"].values,
        traj_speed_arr=cand_speed,
        traj_dt_arr=cand_dt,
        traj_dist_arr=cand_dist,
        traj_bearing_arr=cand_bearing,
        is_zero_speed_arr=cand_zero_spd
    )
    
    # 8. Assemble final candidate table
    n_final_cands = len(df_pruned)
    base_cands_df = pd.DataFrame({
        "trip_id": [trip_id] * n_final_cands,
        "obs_id": cand_obs_ids.astype(int),
        "candidate_id": np.arange(n_final_cands, dtype=int),
        "arc_id": df_pruned["arc_id"].values.astype(int),
        "snapped_lon": df_pruned["snapped_lon"].values,
        "snapped_lat": df_pruned["snapped_lat"].values
    })
    
    final_candidates = pd.concat([base_cands_df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)
    
    return CandidateBatch(
        observations=obs_df,
        candidates=final_candidates,
        schema_version=schema_version
    )
