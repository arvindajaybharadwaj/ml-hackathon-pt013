from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


TRIP_ID_RE = re.compile(r"^[0-9]{8}$")


@dataclass
class TripData:
    trip_id: str
    partition: str
    trip_dir: Path
    track: pd.DataFrame
    nodes_lonlat: np.ndarray
    arcs: np.ndarray


class TripRepository:
    """Loads inference inputs only. It deliberately never loads .route labels."""

    def __init__(self, data_root: Path, partition: str):
        self.data_root = Path(data_root).resolve()
        self.partition = partition
        self.partition_dir = (self.data_root / partition).resolve()
        if not self.partition_dir.is_relative_to(self.data_root):
            raise ValueError("Partition path is outside the configured data root")

    def list_trip_ids(self) -> list[str]:
        if not self.partition_dir.exists():
            return []
        return sorted(
            p.name for p in self.partition_dir.iterdir()
            if p.is_dir() and TRIP_ID_RE.fullmatch(p.name)
        )

    def _trip_dir(self, trip_id: str) -> Path:
        if not TRIP_ID_RE.fullmatch(str(trip_id)):
            raise ValueError("trip_id must contain exactly eight digits")
        path = (self.partition_dir / str(trip_id)).resolve()
        if not path.is_relative_to(self.partition_dir):
            raise ValueError("Invalid trip path")
        if not path.is_dir():
            raise FileNotFoundError(f"Trip {trip_id} was not found in partition {self.partition}")
        return path

    def load_trip(self, trip_id: str) -> TripData:
        """Read .track, .nodes and .arcs; reference .route is not accessed here."""
        trip_dir = self._trip_dir(trip_id)
        stem = str(trip_id)
        track_path = trip_dir / f"{stem}.track"
        nodes_path = trip_dir / f"{stem}.nodes"
        arcs_path = trip_dir / f"{stem}.arcs"
        for path in (track_path, nodes_path, arcs_path):
            if not path.is_file():
                raise FileNotFoundError(f"Required trip file is missing: {path.name}")

        track = pd.read_csv(track_path, sep=r"\s+", header=None, names=["longitude", "latitude", "timestamp"])
        nodes = pd.read_csv(nodes_path, sep=r"\s+", header=None, names=["longitude", "latitude"])
        arcs = pd.read_csv(arcs_path, sep=r"\s+", header=None, names=["from_node_id", "to_node_id"])

        track = track.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
        nodes = nodes.apply(pd.to_numeric, errors="coerce")
        arcs = arcs.apply(pd.to_numeric, errors="coerce")
        if track.empty or nodes.empty or arcs.empty:
            raise ValueError(f"Trip {trip_id} has an empty or invalid required file")

        node_array = nodes.to_numpy(dtype=float)
        arc_array = arcs.to_numpy(dtype=np.int64)
        if not np.isfinite(node_array).all():
            raise ValueError(f"Trip {trip_id} contains invalid node coordinates")
        if (arc_array < 0).any() or (arc_array >= len(node_array)).any():
            raise ValueError(f"Trip {trip_id} contains arc endpoints outside the node table")
        if not np.isfinite(track[["longitude", "latitude", "timestamp"]].to_numpy()).all():
            raise ValueError(f"Trip {trip_id} contains invalid GPS values")
        if not track.longitude.between(-180, 180).all() or not track.latitude.between(-90, 90).all():
            raise ValueError(f"Trip {trip_id} contains geographically invalid coordinates")

        # GPS records should be chronological. Stable sorting is safe if a file has small ordering issues.
        track = track.sort_values("timestamp", kind="stable").reset_index(drop=True)
        return TripData(
            trip_id=str(trip_id), partition=self.partition, trip_dir=trip_dir,
            track=track, nodes_lonlat=node_array, arcs=arc_array,
        )

    def metadata(self, trip_id: str) -> dict:
        trip = self.load_trip(trip_id)
        return {
            "trip_id": trip.trip_id,
            "partition": trip.partition,
            "gps_observations": int(len(trip.track)),
            "road_nodes": int(len(trip.nodes_lonlat)),
            "road_arcs": int(len(trip.arcs)),
            "bounds": {
                "min_longitude": float(trip.track.longitude.min()),
                "max_longitude": float(trip.track.longitude.max()),
                "min_latitude": float(trip.track.latitude.min()),
                "max_latitude": float(trip.track.latitude.max()),
            },
        }


class ReferenceRouteRepository:
    """Separate label store. Only the evaluation service should use this class."""

    def __init__(self, trip_repository: TripRepository):
        self.trip_repository = trip_repository

    def load_reference_route(self, trip_id: str) -> list[int]:
        trip_dir = self.trip_repository._trip_dir(trip_id)
        route_path = trip_dir / f"{trip_id}.route"
        if not route_path.is_file():
            raise FileNotFoundError(f"Reference route is not available for trip {trip_id}")
        values = pd.read_csv(route_path, sep=r"\s+", header=None).iloc[:, 0]
        route = pd.to_numeric(values, errors="coerce").dropna().astype(int).tolist()
        return route
