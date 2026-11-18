from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .network import RoadNetwork


@dataclass(frozen=True)
class HMMConfig:
    """Configurable transition model parameters (distances are metres)."""

    transition_sigma_base_m: float = 30.0
    transition_sigma_fraction: float = 0.5
    missing_state_penalty: float = 6.0
    unreachable_penalty: float = 12.0
    max_speed_mps: float = 70.0
    gap_scale_after_s: float = 10.0

    def __post_init__(self):
        positive = ("transition_sigma_base_m", "max_speed_mps", "gap_scale_after_s")
        nonnegative = ("transition_sigma_fraction", "missing_state_penalty", "unreachable_penalty")
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be greater than zero")
        for name in nonnegative:
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")


@dataclass
class DecodeResult:
    arc_ids: list[int | None]
    candidate_scores: list[float]
    log_probability: float
    missing_observations: int
    diagnostics: dict = field(default_factory=dict)


class HMMViterbiDecoder:
    """Variable-state HMM decoder over candidate directed road arcs.

    Candidate model scores are normalised within each observation to form
    emission probabilities. Transition likelihood compares directed-network
    distance with GPS displacement, with a wider tolerance for long time gaps.
    """

    def __init__(self, config: HMMConfig | None = None):
        self.config = config or HMMConfig()

    def decode(
        self,
        observations: pd.DataFrame,
        candidates: pd.DataFrame,
        network: RoadNetwork,
        score_column: str = "model_score",
    ) -> DecodeResult:
        required_obs = {"longitude", "latitude", "timestamp"}
        missing_obs_cols = required_obs - set(observations.columns)
        if missing_obs_cols:
            raise ValueError(f"Observations missing columns: {sorted(missing_obs_cols)}")
        required_candidate_cols = {"obs_index", "arc_id", "fraction"}
        missing_candidate_cols = required_candidate_cols - set(candidates.columns)
        if not candidates.empty and missing_candidate_cols:
            raise ValueError(f"Candidates missing columns: {sorted(missing_candidate_cols)}")

        n_obs = len(observations)
        if n_obs == 0:
            if not candidates.empty:
                raise ValueError("Candidates were supplied for an empty observation sequence")
            return DecodeResult([], [], 0.0, 0, {"transition_count": 0, "unreachable_transitions": 0,
                "fallback_transitions": 0, "long_gaps": 0, "candidate_counts": []})
        if not candidates.empty:
            obs_indices = pd.to_numeric(candidates["obs_index"], errors="coerce")
            if obs_indices.isna().any() or not obs_indices.between(0, n_obs - 1).all():
                raise ValueError("Candidate obs_index is outside the observation sequence")

        grouped = {int(k): v for k, v in candidates.groupby("obs_index", sort=False)} if not candidates.empty else {}
        states: list[list[dict]] = []
        missing_count = 0
        candidate_counts: list[int] = []
        for obs_idx in range(n_obs):
            group = grouped.get(obs_idx)
            if group is None or group.empty:
                states.append([{"arc_id": None, "fraction": 0.0, "log_emit": 0.0, "raw_score": 0.0}])
                missing_count += 1
                candidate_counts.append(0)
                continue
            if score_column in group:
                scores = pd.to_numeric(group[score_column], errors="coerce").to_numpy(dtype=float)
            else:
                scores = np.ones(len(group), dtype=float)
            scores = np.nan_to_num(scores, nan=1e-9, posinf=1.0, neginf=1e-9)
            scores = np.clip(scores, 1e-9, None)
            scores /= scores.sum()
            current_states = []
            for (_, row), probability in zip(group.iterrows(), scores):
                arc_id = int(row.arc_id)
                if arc_id < 0 or (hasattr(network, "arcs") and arc_id >= len(network.arcs)):
                    raise ValueError(f"Candidate arc_id {arc_id} is outside the road network")
                current_states.append({
                    "arc_id": arc_id,
                    "fraction": float(np.clip(row.fraction, 0.0, 1.0)),
                    "log_emit": float(np.log(max(probability, 1e-12))),
                    "raw_score": float(probability),
                })
            states.append(current_states)
            candidate_counts.append(len(current_states))

        dp: list[np.ndarray] = [np.array([s["log_emit"] for s in states[0]], dtype=float)]
        backpointers: list[np.ndarray] = [np.full(len(states[0]), -1, dtype=int)]
        transition_count = 0
        unreachable_transitions = 0
        fallback_transitions = 0
        long_gaps = 0

        for t in range(1, n_obs):
            gps_distance, delta_t = _observation_step(observations, t)
            if delta_t >= self.config.gap_scale_after_s:
                long_gaps += 1
            step_scores = np.full(len(states[t]), -np.inf, dtype=float)
            step_back = np.full(len(states[t]), -1, dtype=int)
            cutoff = max(
                250.0,
                gps_distance * 4.0 + 150.0,
                delta_t * self.config.max_speed_mps * 1.5,
            )
            for j, curr in enumerate(states[t]):
                best_value = -np.inf
                best_i = -1
                for i, prev in enumerate(states[t - 1]):
                    transition_count += 1
                    transition, unreachable = self._transition_log_probability(
                        prev, curr, gps_distance, delta_t, network, cutoff
                    )
                    if unreachable:
                        unreachable_transitions += 1
                    value = dp[t - 1][i] + transition
                    if value > best_value:
                        best_value, best_i = value, i
                step_scores[j] = best_value + curr["log_emit"]
                step_back[j] = best_i

            if not np.isfinite(step_scores).any():
                fallback_transitions += 1
                step_scores = np.array([
                    dp[t - 1].max() - self.config.unreachable_penalty + s["log_emit"]
                    for s in states[t]
                ])
                step_back[:] = int(np.argmax(dp[t - 1]))
            dp.append(step_scores)
            backpointers.append(step_back)

        final_idx = int(np.argmax(dp[-1]))
        best_log_probability = float(dp[-1][final_idx])
        selected = [final_idx]
        for t in range(n_obs - 1, 0, -1):
            previous = int(backpointers[t][selected[-1]])
            if previous < 0:
                previous = int(np.argmax(dp[t - 1]))
            selected.append(previous)
        selected.reverse()
        arc_ids = [states[t][idx]["arc_id"] for t, idx in enumerate(selected)]
        scores = [states[t][idx]["raw_score"] for t, idx in enumerate(selected)]
        return DecodeResult(
            arc_ids, scores, best_log_probability, missing_count,
            {
                "transition_count": transition_count,
                "unreachable_transitions": unreachable_transitions,
                "fallback_transitions": fallback_transitions,
                "long_gaps": long_gaps,
                "candidate_counts": candidate_counts,
                "mean_candidates_per_observation": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
            },
        )

    def _transition_log_probability(self, prev, curr, gps_distance, delta_t, network, cutoff):
        if prev["arc_id"] is None and curr["arc_id"] is None:
            return 0.0, False
        if prev["arc_id"] is None or curr["arc_id"] is None:
            return -self.config.missing_state_penalty, False
        route_distance = network.route_distance(
            prev["arc_id"], prev["fraction"], curr["arc_id"], curr["fraction"], cutoff
        )
        if route_distance is None:
            return -self.config.unreachable_penalty, True
        gap_factor = max(1.0, delta_t / self.config.gap_scale_after_s)
        sigma = (
            self.config.transition_sigma_base_m
            + self.config.transition_sigma_fraction * max(gps_distance, 1.0)
        ) * np.sqrt(gap_factor)
        mismatch = abs(route_distance - gps_distance)
        return -0.5 * (mismatch / max(sigma, 1e-6)) ** 2, False


def _observation_step(observations: pd.DataFrame, index: int) -> tuple[float, float]:
    previous = observations.iloc[index - 1]
    current = observations.iloc[index]
    delta_t = max(float(current.timestamp - previous.timestamp), 0.001)
    lon1, lat1 = np.radians([previous.longitude, previous.latitude])
    lon2, lat2 = np.radians([current.longitude, current.latitude])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    distance = 2 * 6371008.8 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return float(distance), float(delta_t)
