from __future__ import annotations

from collections import OrderedDict
import heapq
import itertools
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

    def __init__(self, trip: TripData, cache_size: int = 2048, max_search_nodes: int = 500):
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
        self._cache_size = cache_size * 16
        self.max_search_nodes = int(max_search_nodes)
        # Cache exact source/target shortest-path results. A* with a straight-line
        # admissible heuristic avoids exploring a large radius around every source.
        self._distance_cache: OrderedDict[tuple[int, int], float | None] = OrderedDict()
        self._failure_cutoffs = {}
        self._path_cache = OrderedDict()

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
        source, target = int(source), int(target)
        if source == target:
            return 0.0
        if source < 0 or target < 0 or source >= len(self.node_xy) or target >= len(self.node_xy):
            return None
        straight_line = float(np.linalg.norm(self.node_xy[source] - self.node_xy[target]))
        if straight_line > cutoff_m:
            return None
        key = (source, target)
        if key in self._distance_cache:
            value = self._distance_cache[key]
            self._distance_cache.move_to_end(key)
            if value is not None:
                return value if value <= cutoff_m else None
            if cutoff_m <= self._failure_cutoffs.get(key, -1):
                return None
        # Bounded A*: straight-line distance is admissible because each edge
        # weight is its projected Euclidean length. The expansion cap prevents
        # disconnected candidates from causing an unbounded graph search.
        counter = itertools.count()
        queue = [(straight_line, 0.0, next(counter), source)]
        best = {source: 0.0}
        expanded = 0
        value = None
        while queue and expanded < self.max_search_nodes:
            _, distance, _, node = heapq.heappop(queue)
            if distance != best.get(node):
                continue
            if node == target:
                value = float(distance)
                break
            expanded += 1
            for neighbour, edge_data in self.graph[node].items():
                for attrs in edge_data.values():
                    edge_weight = float(attrs.get("weight", 0.0))
                    candidate_distance = distance + edge_weight
                    if candidate_distance > cutoff_m or candidate_distance >= best.get(neighbour, float("inf")):
                        continue
                    best[neighbour] = candidate_distance
                    heuristic = float(np.linalg.norm(self.node_xy[neighbour] - self.node_xy[target]))
                    heapq.heappush(queue, (candidate_distance + heuristic, candidate_distance,
                                           next(counter), int(neighbour)))
        self._distance_cache[key] = value
        if value is None:
            self._failure_cutoffs[key] = cutoff_m
        else:
            self._failure_cutoffs.pop(key, None)
        if len(self._distance_cache) > self._cache_size:
            expired, _ = self._distance_cache.popitem(last=False)
            self._failure_cutoffs.pop(expired, None)
        return value if value is not None and value <= cutoff_m else None

    def shortest_path_arcs(self, source: int, target: int, cutoff_m: float):
        """Bounded A* returning actual edge IDs for route geometry, not just distance."""
        if source == target:
            return []
        key = (source, target, float(cutoff_m))
        if key in self._path_cache:
            return self._path_cache[key]
        queue = [(float(np.linalg.norm(self.node_xy[source]-self.node_xy[target])), 0., source)]
        best, parents, expanded = {source: 0.}, {}, 0
        result = None
        while queue and expanded < self.max_search_nodes:
            _, cost, node = heapq.heappop(queue)
            if cost != best.get(node):
                continue
            if node == target:
                result = []
                while node != source:
                    node, arc = parents[node]
                    result.append(arc)
                result.reverse()
                break
            expanded += 1
            for neighbour, edges in self.graph[node].items():
                for attrs in edges.values():
                    candidate = cost + float(attrs['weight'])
                    if candidate > cutoff_m or candidate >= best.get(neighbour, float('inf')):
                        continue
                    best[neighbour] = candidate
                    parents[neighbour] = (node, int(attrs['arc_id']))
                    heuristic = float(np.linalg.norm(self.node_xy[neighbour]-self.node_xy[target]))
                    heapq.heappush(queue, (candidate+heuristic, candidate, int(neighbour)))
        self._path_cache[key] = result
        if len(self._path_cache) > 4096:
            self._path_cache.popitem(last=False)
        return result
