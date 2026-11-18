from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import networkx as nx
import numpy as np
from pyproj import CRS, Transformer

from .data import TripData


@dataclass(frozen=True)
class Projector:
    forward: Transformer
    inverse: Transformer

    @classmethod
    def for_coordinates(cls, lon: np.ndarray, lat: np.ndarray) -> "Projector":
        lon0 = float(np.nanmedian(lon))
        lat0 = float(np.nanmedian(lat))
        local = CRS.from_proj4(
            f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
        )
        wgs84 = CRS.from_epsg(4326)
        return cls(
            forward=Transformer.from_crs(wgs84, local, always_xy=True),
            inverse=Transformer.from_crs(local, wgs84, always_xy=True),
        )

    def project(self, lon, lat):
        return self.forward.transform(lon, lat)

    def unproject(self, x, y):
        return self.inverse.transform(x, y)


class RoadNetwork:
    """Directed road graph with metric edge lengths and cached bounded shortest paths."""

    def __init__(self, trip: TripData, cache_size: int = 2048):
        self.trip = trip
        self.projector = Projector.for_coordinates(
            np.concatenate([trip.nodes_lonlat[:, 0], trip.track.longitude.to_numpy()]),
            np.concatenate([trip.nodes_lonlat[:, 1], trip.track.latitude.to_numpy()]),
        )
        self.node_x, self.node_y = self.projector.project(
            trip.nodes_lonlat[:, 0], trip.nodes_lonlat[:, 1]
        )
        self.node_xy = np.column_stack([self.node_x, self.node_y])
        self.arcs = trip.arcs
        starts = self.arcs[:, 0]
        ends = self.arcs[:, 1]
        delta = self.node_xy[ends] - self.node_xy[starts]
        self.arc_lengths = np.linalg.norm(delta, axis=1)
        self.arc_bearings = (np.degrees(np.arctan2(delta[:, 0], delta[:, 1])) + 360.0) % 360.0
        self.graph = nx.MultiDiGraph()
        self.graph.add_nodes_from(range(len(trip.nodes_lonlat)))
        for arc_id, (u, v) in enumerate(self.arcs):
            length = float(self.arc_lengths[arc_id])
            self.graph.add_edge(int(u), int(v), key=int(arc_id), weight=max(length, 0.01), arc_id=int(arc_id))
        self._cache_size = cache_size
        self._distance_cache: OrderedDict[tuple[int, int], dict[int, float]] = OrderedDict()

    def edge_coordinates(self, arc_id: int) -> list[list[float]]:
        u, v = self.arcs[int(arc_id)]
        return [
            [float(self.trip.nodes_lonlat[u, 0]), float(self.trip.nodes_lonlat[u, 1])],
            [float(self.trip.nodes_lonlat[v, 0]), float(self.trip.nodes_lonlat[v, 1])],
        ]

    def route_distance(
        self,
        previous_arc: int,
        previous_fraction: float,
        current_arc: int,
        current_fraction: float,
        cutoff_m: float,
    ) -> float | None:
        """Distance along directed edges between two candidate positions."""
        previous_arc = int(previous_arc)
        current_arc = int(current_arc)
        prev_fraction = float(np.clip(previous_fraction, 0.0, 1.0))
        curr_fraction = float(np.clip(current_fraction, 0.0, 1.0))
        prev_length = float(self.arc_lengths[previous_arc])
        curr_length = float(self.arc_lengths[current_arc])

        if previous_arc == current_arc:
            if curr_fraction >= prev_fraction:
                return (curr_fraction - prev_fraction) * prev_length
            # Backtracking along a directed arc is not a normal forward transition.
            return None

        prev_end = int(self.arcs[previous_arc, 1])
        curr_start = int(self.arcs[current_arc, 0])
        middle = self.shortest_distance(prev_end, curr_start, cutoff_m)
        if middle is None:
            return None
        return (1.0 - prev_fraction) * prev_length + middle + curr_fraction * curr_length

    def shortest_distance(self, source: int, target: int, cutoff_m: float) -> float | None:
        if source == target:
            return 0.0
        cutoff_bucket = max(100, int(np.ceil(cutoff_m / 100.0) * 100))
        key = (int(source), cutoff_bucket)
        distances = self._distance_cache.get(key)
        if distances is None:
            try:
                distances = nx.single_source_dijkstra_path_length(
                    self.graph, int(source), cutoff=float(cutoff_bucket), weight="weight"
                )
            except (nx.NetworkXError, nx.NodeNotFound):
                distances = {}
            self._distance_cache[key] = distances
            if len(self._distance_cache) > self._cache_size:
                self._distance_cache.popitem(last=False)
        else:
            self._distance_cache.move_to_end(key)
        value = distances.get(int(target))
        return float(value) if value is not None else None
