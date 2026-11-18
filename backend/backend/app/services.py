from __future__ import annotations

from time import perf_counter
from functools import lru_cache
import numpy as np
from sklearn.metrics import f1_score

from .candidates import CandidateProvider
from .data import ReferenceRouteRepository, TripRepository
from .hmm import HMMViterbiDecoder, _observation_step
from .network import RoadNetwork
from .predictor import CandidatePredictor


class PredictionService:
    def __init__(self, trips: TripRepository, candidate_provider: CandidateProvider,
                 predictor: CandidatePredictor, decoder: HMMViterbiDecoder):
        self.trips = trips
        self.candidate_provider = candidate_provider
        self.predictor = predictor
        self.decoder = decoder

    @lru_cache(maxsize=4)
    def predict(self, trip_id: str) -> dict:
        started = perf_counter()
        trip = self.trips.load_trip(trip_id)
        network = RoadNetwork(trip)

        stage = perf_counter()
        candidates = self.candidate_provider.generate(trip, network)
        candidate_generation_ms = (perf_counter() - stage) * 1000.0

        stage = perf_counter()
        scores = self.predictor.score_candidates(candidates)
        candidates = candidates.copy()
        candidates["model_score"] = scores
        model_scoring_ms = (perf_counter() - stage) * 1000.0

        stage = perf_counter()
        decoded = self.decoder.decode(trip.track, candidates, network)
        hmm_decoding_ms = (perf_counter() - stage) * 1000.0

        grouped_candidates = {int(k): v for k, v in candidates.groupby("obs_index", sort=False)} if not candidates.empty else {}
        classifier_only = []
        for obs_index in range(len(trip.track)):
            group = grouped_candidates.get(obs_index)
            if group is None or group.empty:
                classifier_only.append(None)
            else:
                classifier_only.append(int(group.loc[group.model_score.idxmax(), "arc_id"]))
        route_continuity = _route_continuity(decoded.arc_ids, network, trip.track, self.decoder)
        classifier_continuity = _route_continuity(classifier_only, network, trip.track, self.decoder)

        route_ids = expand_route(decoded.arc_ids, network, trip.track, self.decoder)
        classifier_route_ids = expand_route(classifier_only, network, trip.track, self.decoder)
        route_segments = []
        segment = []
        for arc_id in route_ids:
            if arc_id is None:
                if segment:
                    route_segments.append(segment)
                    segment = []
                continue
            edge = network.edge_coordinates(arc_id)
            if not segment:
                segment.extend(edge)
            elif segment[-1] == edge[0]:
                segment.append(edge[1])
            else:
                route_segments.append(segment)
                segment = list(edge)
        if segment:
            route_segments.append(segment)

        matches = []
        for obs_index, arc_id in enumerate(decoded.arc_ids):
            group = grouped_candidates.get(obs_index)
            selected = group[group.arc_id == arc_id] if arc_id is not None and group is not None else candidates.iloc[0:0]
            row = selected.iloc[0] if not selected.empty else None
            matches.append({
                "obs_index": obs_index,
                "arc_id": int(arc_id) if arc_id is not None else None,
                "candidate_score": float(decoded.candidate_scores[obs_index]),
                "distance_m": float(row.distance_m) if row is not None and "distance_m" in row else None,
            })
        elapsed = (perf_counter() - started) * 1000.0
        return {
            "trip_id": trip.trip_id,
            "partition": trip.partition,
            "prediction_mode": self.predictor.mode,
            "predictor": self.predictor.name,
            "candidate_provider": type(self.candidate_provider).__name__,
            "gps_observations": int(len(trip.track)),
            "candidate_rows": int(len(candidates)),
            "matched_observations": int(sum(x is not None for x in decoded.arc_ids)),
            "missing_observations": int(decoded.missing_observations),
            "predicted_arc_ids": decoded.arc_ids,
            "classifier_only_arc_ids": classifier_only,
            "route_continuity": round(route_continuity, 6),
            "classifier_only_route_continuity": round(classifier_continuity, 6),
            # Legacy field contains one continuous piece; route_segments is canonical.
            "route_coordinates": max(route_segments, key=len, default=[]),
            "route_segments": route_segments,
            "predicted_route_arc_ids": route_ids,
            "classifier_route_arc_ids": classifier_route_ids,
            "matches": matches,
            "viterbi_log_probability": decoded.log_probability,
            "decoder_diagnostics": decoded.diagnostics,
            "timing_breakdown_ms": {
                "candidate_generation": round(candidate_generation_ms, 3),
                "model_scoring": round(model_scoring_ms, 3),
                "hmm_decoding": round(hmm_decoding_ms, 3),
                "total": round(elapsed, 3),
            },
            "processing_time_ms": round(elapsed, 2),
            "coordinate_order": "[longitude, latitude]",
        }


class EvaluationService:
    """Evaluation is separate from inference; reference labels are read only here."""

    def __init__(self, prediction_service: PredictionService, references: ReferenceRouteRepository):
        self.prediction_service = prediction_service
        self.references = references

    def evaluate(self, trip_id: str) -> dict:
        prediction = self.prediction_service.predict(trip_id)
        # Ground truth is deliberately loaded only after prediction has completed.
        reference = self.references.load_reference_route(trip_id)
        predicted = _compress_consecutive([x for x in prediction["predicted_route_arc_ids"] if x is not None])
        reference_compact = _compress_consecutive(reference)
        classifier_only = _compress_consecutive([
            x for x in prediction["classifier_route_arc_ids"] if x is not None
        ])
        edit_distance = _levenshtein(predicted, reference_compact)
        classifier_edit_distance = _levenshtein(classifier_only, reference_compact)
        denominator = max(len(predicted), len(reference_compact), 1)
        lcs_length = _lcs_length(predicted, reference_compact)
        ref_set, pred_set = set(reference_compact), set(predicted)
        intersection = len(ref_set & pred_set)
        classifier_intersection = len(ref_set & set(classifier_only))
        classifier_precision = classifier_intersection / len(set(classifier_only)) if classifier_only else 0.0
        classifier_recall = classifier_intersection / len(ref_set) if ref_set else 0.0
        classifier_f1 = (2 * classifier_precision * classifier_recall / (classifier_precision + classifier_recall)
                         if classifier_precision + classifier_recall else 0.0)
        precision = intersection / len(pred_set) if pred_set else 0.0
        recall = intersection / len(ref_set) if ref_set else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        aligned = self.references.load_aligned_labels(trip_id)
        if aligned.duplicated('obs_id').any() or set(aligned.obs_id) != set(range(prediction['gps_observations'])):
            raise ValueError('Aligned evaluation labels must cover every original observation exactly once')
        aligned = aligned.sort_values('obs_id')
        valid = (aligned.alignment_valid & aligned.reference_arc_id.notna()).to_numpy(dtype=bool)
        truth = aligned.loc[valid, 'reference_arc_id'].to_numpy(dtype=int)
        pointwise = {'alignment_coverage': float(valid.mean()), 'evaluated_observations': int(valid.sum())}
        for name, field in [('classifier', 'classifier_only_arc_ids'), ('hmm', 'predicted_arc_ids')]:
            values = np.array([-1 if x is None else x for x in prediction[field]])[valid]
            pointwise[name + '_correct_observations'] = int((values == truth).sum())
            pointwise[name + '_top1_accuracy'] = float((values == truth).mean()) if len(truth) else None
            pointwise[name + '_observation_arc_macro_f1'] = float(f1_score(truth, values, average='macro', zero_division=0)) if len(truth) else None
        return {
            **pointwise,
            "trip_id": trip_id,
            "prediction_mode": prediction["prediction_mode"],
            "predicted_sequence_length": len(predicted),
            "reference_sequence_length": len(reference_compact),
            "sequence_edit_similarity": round(1.0 - edit_distance / denominator, 6),
            "classifier_only_sequence_edit_similarity": round(1.0 - classifier_edit_distance / max(len(classifier_only), len(reference_compact), 1), 6),
            "hmm_route_continuity": prediction["route_continuity"],
            "classifier_only_route_continuity": prediction["classifier_only_route_continuity"],
            "classifier_only_arc_set_f1": round(classifier_f1, 6),
            "lcs_recall_against_reference": round(lcs_length / max(len(reference_compact), 1), 6),
            "arc_set_precision": round(precision, 6),
            "arc_set_recall": round(recall, 6),
            "arc_set_f1": round(f1, 6),
            "timing_breakdown_ms": prediction["timing_breakdown_ms"],
            "decoder_diagnostics": prediction["decoder_diagnostics"],
            "note": (
                "Sequence/set metrics compare expanded road paths with the reference route. "
                "Pointwise metrics use separately aligned observation labels; inspect alignment coverage."
            ),
        }


def _compress_consecutive(values: list[int]) -> list[int]:
    result = []
    for value in values:
        if not result or result[-1] != value:
            result.append(int(value))
    return result


def _levenshtein(a: list[int], b: list[int]) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def _lcs_length(a: list[int], b: list[int]) -> int:
    previous = [0] * (len(b) + 1)
    for x in a:
        current = [0]
        for j, y in enumerate(b, 1):
            current.append(previous[j - 1] + 1 if x == y else max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def _route_continuity(arc_ids: list[int | None], network, observations, decoder) -> float:
    """Fraction of consecutive matched arc predictions with direct directed continuity.

    A pair is continuous when it remains on the same arc or the previous arc's
    end node is the current arc's start node. This intentionally measures direct
    adjacency (not arbitrary multi-hop reachability), so it is fast, deterministic
    and has a precise interpretation for the report.
    """
    pairs = list(zip(arc_ids, arc_ids[1:]))
    if not pairs:
        return 0.0
    connected = 0
    for previous, current in pairs:
        if previous is None or current is None:
            continue
        if previous == current or int(network.arcs[int(previous), 1]) == int(network.arcs[int(current), 0]):
            connected += 1
    return connected / len(pairs)


def expand_route(arc_ids, network, observations, decoder):
    """Insert bounded directed shortest paths; preserve explicit breaks when unreachable."""
    result = []
    previous = None
    for index, current in enumerate(arc_ids):
        if current is None:
            if result and result[-1] is not None:
                result.append(None)
            previous = None
            continue
        if previous is not None and current != previous:
            distance, dt = _observation_step(observations, index)
            cutoff = max(250., distance * 4 + 150, dt * decoder.config.max_speed_mps * 1.5)
            bridge = network.shortest_path_arcs(int(network.arcs[previous, 1]),
                                                int(network.arcs[current, 0]), cutoff)
            if bridge is None:
                result.append(None)
            else:
                result.extend(bridge)
        if previous != current:
            result.append(int(current))
        previous = current
    return result
