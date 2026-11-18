import pandas as pd

from backend.app.hmm import HMMConfig, HMMViterbiDecoder


class TinyNetwork:
    def route_distance(self, previous_arc, previous_fraction, current_arc, current_fraction, cutoff_m):
        # 1 -> 2 is connected; 2 -> 1 is not.
        if previous_arc == current_arc:
            return 0.0 if current_fraction >= previous_fraction else None
        if previous_arc == 1 and current_arc == 2:
            return 20.0
        return None


def test_viterbi_uses_transition_coherence():
    observations = pd.DataFrame({
        "longitude": [0.0, 0.0001],
        "latitude": [0.0, 0.0],
        "timestamp": [1.0, 2.0],
    })
    candidates = pd.DataFrame([
        {"obs_index": 0, "arc_id": 1, "fraction": 0.2, "model_score": 0.9},
        {"obs_index": 0, "arc_id": 2, "fraction": 0.4, "model_score": 0.1},
        # At t=1, arc 2 has the higher local score, but arc 1 -> 2 is the valid transition.
        {"obs_index": 1, "arc_id": 2, "fraction": 0.5, "model_score": 0.6},
        {"obs_index": 1, "arc_id": 1, "fraction": 0.7, "model_score": 0.4},
    ])
    decoder = HMMViterbiDecoder(HMMConfig(transition_sigma_base_m=10, transition_sigma_fraction=0.1))
    result = decoder.decode(observations, candidates, TinyNetwork())
    assert len(result.arc_ids) == 2
    assert result.arc_ids[0] == 1
    assert result.arc_ids[1] == 2
    assert result.missing_observations == 0


def test_viterbi_handles_missing_candidate_observation():
    observations = pd.DataFrame({
        "longitude": [0.0, 0.0001, 0.0002],
        "latitude": [0.0, 0.0, 0.0],
        "timestamp": [1.0, 2.0, 3.0],
    })
    candidates = pd.DataFrame([
        {"obs_index": 0, "arc_id": 1, "fraction": 0.1, "model_score": 1.0},
        {"obs_index": 2, "arc_id": 2, "fraction": 0.5, "model_score": 1.0},
    ])
    result = HMMViterbiDecoder().decode(observations, candidates, TinyNetwork())
    assert len(result.arc_ids) == 3
    assert result.arc_ids[1] is None
    assert result.missing_observations == 1
