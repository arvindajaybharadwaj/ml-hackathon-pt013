from __future__ import annotations

from pathlib import Path
import json
from typing import Protocol

import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

from .data import TripData
from .network import RoadNetwork


REQUIRED_CANDIDATE_COLUMNS = {"obs_index", "arc_id", "distance_m", "fraction"}


class CandidateProvider(Protocol):
    def generate(self, trip: TripData, network: RoadNetwork) -> pd.DataFrame: ...


class PreparedCandidateProvider:
    """Reads Person 1's prepared candidate table: <trip_id>.csv or .parquet."""

    def __init__(self, candidate_dir: Path):
        self.candidate_dir = Path(candidate_dir)
        self.schema_version = None
        for parent in [self.candidate_dir, *list(self.candidate_dir.parents)[:2]]:
            schema = parent / 'feature_schema.json'
            if schema.is_file():
                self.schema_version = json.loads(schema.read_text())['schema_version']
                break

    def available_trip_ids(self) -> set[str]:
        combined = self.candidate_dir / 'candidates.parquet'
        if combined.is_file():
            return set(pd.read_parquet(combined, columns=['trip_id']).trip_id.astype(str))
        return {p.stem for ext in ('*.csv', '*.parquet') for p in self.candidate_dir.glob(ext)}

    def generate(self, trip: TripData, network: RoadNetwork) -> pd.DataFrame:
        csv_path = self.candidate_dir / f"{trip.trip_id}.csv"
        parquet_path = self.candidate_dir / f"{trip.trip_id}.parquet"
        combined = self.candidate_dir / 'candidates.parquet'
        if combined.is_file():
            df = pd.read_parquet(combined, filters=[('trip_id', '=', trip.trip_id)])
        elif parquet_path.is_file():
            df = pd.read_parquet(parquet_path)
        elif csv_path.is_file():
            df = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(
                f"No prepared candidates for {trip.trip_id}; expected {csv_path.name} or {parquet_path.name}"
            )
        if 'trip_id' in df and not df.trip_id.eq(trip.trip_id).all():
            raise ValueError('Prepared candidates contain another trip')
        # Preserve the real ML feature names; add decoder-specific aliases only.
        for target, source in [('obs_index', 'obs_id'), ('distance_m', 'distance_to_segment_m'),
                               ('fraction', 'projection_fraction')]:
            if target not in df and source in df:
                df[target] = df[source]
        missing = REQUIRED_CANDIDATE_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Person 1 candidate table is missing columns: {sorted(missing)}")
        result = _validate_candidates(df, trip)
        result.attrs['schema_version'] = self.schema_version
        return result


class NearestArcCandidateProvider:
    """Development fallback only; replace with Person 1's candidate generator for final work."""

    def __init__(self, radius_m: float = 100.0, max_candidates: int = 8):
        self.radius_m = float(radius_m)
        self.max_candidates = int(max_candidates)

    def generate(self, trip: TripData, network: RoadNetwork) -> pd.DataFrame:
        lines = [
            LineString([network.node_xy[int(u)], network.node_xy[int(v)]])
            for u, v in trip.arcs
        ]
        tree = STRtree(lines)
        rows: list[dict] = []
        track = trip.track
        px, py = network.projector.project(track.longitude.to_numpy(), track.latitude.to_numpy())
        points = [Point(float(x), float(y)) for x, y in zip(px, py)]
        headings = _headings(track.longitude.to_numpy(), track.latitude.to_numpy())
        dt = np.diff(track.timestamp.to_numpy(), prepend=track.timestamp.iloc[0]).astype(float)
        gps_step = np.zeros(len(track), dtype=float)
        if len(track) > 1:
            gps_step[1:] = _haversine_array(
                track.longitude.to_numpy()[:-1], track.latitude.to_numpy()[:-1],
                track.longitude.to_numpy()[1:], track.latitude.to_numpy()[1:]
            )
        for obs_index, point in enumerate(points):
            indices = tree.query(point.buffer(self.radius_m))
            if len(indices) == 0:
                nearest = tree.nearest(point)
                indices = np.asarray([nearest], dtype=int)
            options = []
            for arc_idx in np.asarray(indices, dtype=int).tolist():
                line = lines[arc_idx]
                length = max(float(line.length), 0.01)
                along = float(line.project(point))
                nearest_point = line.interpolate(along)
                distance = float(point.distance(nearest_point))
                fraction = float(np.clip(along / length, 0.0, 1.0))
                if distance <= self.radius_m:
                    options.append((distance, int(arc_idx), fraction, length))
            if not options:
                nearest_idx = int(tree.nearest(point))
                line = lines[nearest_idx]
                length = max(float(line.length), 0.01)
                along = float(line.project(point))
                nearest_point = line.interpolate(along)
                options.append((
                    float(point.distance(nearest_point)), nearest_idx,
                    float(np.clip(along / length, 0.0, 1.0)), length
                ))
            options.sort(key=lambda item: item[0])
            for distance, arc_id, fraction, length in options[: max(1, self.max_candidates)]:
                rows.append({
                    "obs_index": int(obs_index),
                    "arc_id": int(arc_id),
                    "distance_m": distance,
                    "fraction": fraction,
                    "segment_length_m": length,
                    "bearing_deg": float(network.arc_bearings[arc_id]),
                    "observation_heading_deg": float(headings[obs_index]),
                    "gps_step_m": float(gps_step[obs_index]),
                    "delta_t_s": float(dt[obs_index]) if obs_index else 0.0,
                    "longitude": float(track.longitude.iloc[obs_index]),
                    "latitude": float(track.latitude.iloc[obs_index]),
                    "timestamp": float(track.timestamp.iloc[obs_index]),
                    "heading_difference_deg": _angle_diff(
                        float(headings[obs_index]), float(network.arc_bearings[arc_id])
                    ),
                })
        return _validate_candidates(pd.DataFrame(rows), trip)


def _validate_candidates(df: pd.DataFrame, trip: TripData) -> pd.DataFrame:
    result = df.copy()
    missing = REQUIRED_CANDIDATE_COLUMNS - set(result.columns)
    if missing:
        raise ValueError(f"Candidate table is missing columns: {sorted(missing)}")
    for col in ("obs_index", "arc_id", "distance_m", "fraction"):
        result[col] = pd.to_numeric(result[col], errors="coerce")
    result = result.dropna(subset=list(REQUIRED_CANDIDATE_COLUMNS)).copy()
    result["obs_index"] = result.obs_index.astype(int)
    result["arc_id"] = result.arc_id.astype(int)
    if not result.obs_index.between(0, len(trip.track) - 1).all():
        raise ValueError("Candidate obs_index is outside the trip's GPS observations")
    if not result.arc_id.between(0, len(trip.arcs) - 1).all():
        raise ValueError("Candidate arc_id is outside the trip's road network")
    if not np.isfinite(result[["distance_m", "fraction"]].to_numpy(dtype=float)).all():
        raise ValueError("Candidate distances/fractions must be finite")
    result["fraction"] = result.fraction.clip(0.0, 1.0)
    result["distance_m"] = result.distance_m.clip(lower=0.0)
    result = result.sort_values(["obs_index", "distance_m"], kind="stable").reset_index(drop=True)
    return result


def _angle_diff(a: float, b: float) -> float:
    return float(abs((a - b + 180.0) % 360.0 - 180.0))


def _headings(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    result = np.zeros(len(lon), dtype=float)
    if len(lon) < 2:
        return result
    result[:-1] = _bearing_array(lon[:-1], lat[:-1], lon[1:], lat[1:])
    result[-1] = result[-2]
    return result


def _bearing_array(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1); lat1 = np.radians(lat1)
    lon2 = np.radians(lon2); lat2 = np.radians(lat2)
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def _haversine_array(lon1, lat1, lon2, lat2):
    radius = 6371008.8
    lon1 = np.radians(lon1); lat1 = np.radians(lat1)
    lon2 = np.radians(lon2); lat2 = np.radians(lat2)
    dlon = lon2 - lon1; dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
