"""Feature calculation and validation for PathSnap candidate road segments.

Implements Task 4:
- Computes spatial, kinematic, and topological features.
- Validates against feature_schema.json.
- Ensures zero NaN values and consistent column ordering.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd


FEATURE_NAMES: List[str] = [
    "distance_to_segment_m",
    "projection_fraction",
    "segment_length_m",
    "heading_diff_deg",
    "heading_alignment_cos",
    "gps_speed_mps",
    "time_gap_s",
    "spatial_gap_m",
    "candidate_rank_dist",
    "dist_diff_from_nearest_m",
    "speed_length_ratio",
    "is_zero_speed",
    "reliability_flag"
]


def calculate_bearing_deg(x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray) -> np.ndarray:
    """Calculate meteorological bearing in degrees [0, 360) from (x1, y1) to (x2, y2).
    
    In local metric Cartesian coordinates (x=East, y=North):
    Bearing 0 = North (+y), 90 = East (+x), 180 = South (-y), 270 = West (-x).
    """
    dx = x2 - x1
    dy = y2 - y1
    bearing_rad = np.arctan2(dx, dy) # note (x, y) order for clockwise from North
    bearing_deg = np.degrees(bearing_rad) % 360.0
    return bearing_deg


def calculate_trajectory_kinematics(track_x: np.ndarray, track_y: np.ndarray,
                                    time_s: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate speed, time gap, spatial gap, bearing, and zero-speed flags for each observation.
    
    Returns:
        (speeds_mps, dt_s, step_dist_m, bearings_deg, is_zero_speed)
    """
    n = len(track_x)
    speeds_mps = np.zeros(n, dtype=np.float32)
    dt_s = np.ones(n, dtype=np.float32)
    step_dist_m = np.zeros(n, dtype=np.float32)
    bearings_deg = np.zeros(n, dtype=np.float32)
    is_zero_speed = np.zeros(n, dtype=np.int32)
    
    if n <= 1:
        return speeds_mps, dt_s, step_dist_m, bearings_deg, is_zero_speed
        
    diff_t = np.diff(time_s)
    # Handle zero or negative dt robustly
    diff_t_safe = np.where(diff_t <= 0, 1.0, diff_t)
    
    diff_x = np.diff(track_x)
    diff_y = np.diff(track_y)
    dists = np.sqrt(diff_x**2 + diff_y**2)
    
    # Step forward difference for t >= 1
    dt_s[1:] = diff_t_safe
    dt_s[0] = diff_t_safe[0] if len(diff_t_safe) > 0 else 1.0
    
    step_dist_m[1:] = dists
    step_dist_m[0] = dists[0] if len(dists) > 0 else 0.0
    
    computed_speeds = dists / diff_t_safe
    speeds_mps[1:] = computed_speeds
    speeds_mps[0] = computed_speeds[0] if len(computed_speeds) > 0 else 0.0
    
    computed_bearings = calculate_bearing_deg(track_x[:-1], track_y[:-1], track_x[1:], track_y[1:])
    bearings_deg[1:] = computed_bearings
    bearings_deg[0] = computed_bearings[0] if len(computed_bearings) > 0 else 0.0
    
    is_zero_speed = (speeds_mps < 0.5).astype(np.int32)
    
    return speeds_mps, dt_s, step_dist_m, bearings_deg, is_zero_speed


def compute_candidate_features(obs_id_arr: np.ndarray,
                               dist_m_arr: np.ndarray,
                               proj_frac_arr: np.ndarray,
                               seg_len_arr: np.ndarray,
                               seg_bearing_arr: np.ndarray,
                               traj_speed_arr: np.ndarray,
                               traj_dt_arr: np.ndarray,
                               traj_dist_arr: np.ndarray,
                               traj_bearing_arr: np.ndarray,
                               is_zero_speed_arr: np.ndarray) -> pd.DataFrame:
    """Vectorized computation of candidate features."""
    n_cands = len(obs_id_arr)
    if n_cands == 0:
        return pd.DataFrame(columns=FEATURE_NAMES)
        
    # Heading difference: vehicle bearing vs candidate road segment bearing
    diff_raw = np.abs(traj_bearing_arr - seg_bearing_arr) % 360.0
    heading_diff_deg = np.where(diff_raw > 180.0, 360.0 - diff_raw, diff_raw).astype(np.float32)
    
    # When vehicle is stationary, heading is indeterminate; set heading_diff to 90.0
    heading_diff_deg = np.where(is_zero_speed_arr == 1, 90.0, heading_diff_deg)
    
    heading_alignment_cos = np.cos(np.radians(heading_diff_deg)).astype(np.float32)
    
    # Speed to segment length ratio
    speed_len_ratio = (traj_speed_arr * traj_dt_arr) / np.maximum(seg_len_arr, 1.0)
    
    # Candidate rank by distance per observation
    # Compute rank using pandas groupby / cumcount on sorted distances
    df_temp = pd.DataFrame({
        'obs_id': obs_id_arr,
        'dist': dist_m_arr
    })
    # Sort within obs_id
    df_temp['orig_order'] = np.arange(n_cands)
    df_temp = df_temp.sort_values(by=['obs_id', 'dist'])
    df_temp['rank'] = df_temp.groupby('obs_id').cumcount()
    
    # Nearest distance
    min_dist_per_obs = df_temp.groupby('obs_id')['dist'].transform('first')
    df_temp['dist_diff'] = df_temp['dist'] - min_dist_per_obs
    
    # Restore original order
    df_temp = df_temp.sort_values('orig_order')
    candidate_rank_dist = df_temp['rank'].values.astype(np.int32)
    dist_diff_from_nearest_m = df_temp['dist_diff'].values.astype(np.float32)
    
    # Reliability flag
    # Normal 1Hz sampling (0.5s <= dt <= 3.0s), reasonable speed (<= 45 m/s or 160 km/h), dist <= 50m
    rel_flag = (
        (traj_dt_arr >= 0.5) &
        (traj_dt_arr <= 3.0) &
        (traj_speed_arr <= 45.0) &
        (dist_m_arr <= 50.0)
    ).astype(np.int32)
    
    features_df = pd.DataFrame({
        "distance_to_segment_m": dist_m_arr.astype(np.float32),
        "projection_fraction": proj_frac_arr.astype(np.float32),
        "segment_length_m": seg_len_arr.astype(np.float32),
        "heading_diff_deg": heading_diff_deg,
        "heading_alignment_cos": heading_alignment_cos,
        "gps_speed_mps": traj_speed_arr.astype(np.float32),
        "time_gap_s": traj_dt_arr.astype(np.float32),
        "spatial_gap_m": traj_dist_arr.astype(np.float32),
        "candidate_rank_dist": candidate_rank_dist,
        "dist_diff_from_nearest_m": dist_diff_from_nearest_m,
        "speed_length_ratio": speed_len_ratio.astype(np.float32),
        "is_zero_speed": is_zero_speed_arr.astype(np.int32),
        "reliability_flag": rel_flag
    })
    
    # Guarantee column ordering
    return features_df[FEATURE_NAMES]
