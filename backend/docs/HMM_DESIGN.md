# HMM and Viterbi Design – Person 3

## State and observation model

For GPS observation `t`, the hidden-state set is the candidate road-arc set
`S_t = {s_t1, ..., s_tk}` supplied by the candidate provider. A state retains
the original zero-based `arc_id` and a projection fraction along that directed
arc. An explicit `None` state is inserted when an observation has no candidates.

## Emissions

The predictor supplies a non-negative score for each observation/candidate
pair. Scores are clipped away from zero and normalised within each observation:

`P(z_t = s_tj | x_t) = score_tj / sum_k(score_tk)`.

The selected model score is used as the emission. The development fallback
creates a Gaussian-shaped distance score, but is not a trained ML model. Do not
multiply a second distance likelihood into Person 2's probability unless the
team establishes that this does not double-count distance information.

## Transitions

For candidate positions on directed arcs `a` and `b`, the network distance is:

`remaining(a) + shortest_directed_path(end(a), start(b)) + traversed(b)`.

If the same directed arc is used, forward movement is allowed and backward
movement is treated as unreachable. If no directed path is found within the
search bound, a configurable unreachable penalty is used.

Let `d_gps` be the haversine displacement and `d_net` the road-network distance.
The log transition score is a Gaussian mismatch penalty:

`log P(s_t | s_(t-1)) = -0.5 * ((d_net - d_gps) / sigma)^2`.

`sigma = (transition_sigma_base_m + transition_sigma_fraction * max(d_gps, 1)) * sqrt(max(1, delta_t / gap_scale_after_s))`.

A missing state receives a configurable penalty. Longer timestamp gaps widen
the tolerance and increase the network search cutoff. The decoder uses log-space
dynamic programming and backpointers.

## Search bound and diagnostics

Road-network shortest-path search uses bounded A* with projected straight-line
distance as an admissible heuristic. The search expansion cap prevents
unreachable candidate pairs in a large graph from causing unbounded latency.
The cap is currently set to 500 expanded nodes per source-target search in
`RoadNetwork`; adjust only with measured accuracy/latency evidence.

The decoder reports candidate counts, missing observations, attempted
transitions, unreachable transitions, fallback transitions and long GPS gaps.
The API reports per-stage timings for candidate generation, model scoring and
HMM decoding.

## Route continuity metric

The API's `route_continuity` is defined as the fraction of consecutive matched
predicted arc IDs that either remain on the same arc or are directly connected
(`previous_arc.to_node == current_arc.from_node`). It is a direct-adjacency
metric, not a multi-hop reachability metric. The HMM itself can still connect
candidates through intermediate arcs using shortest-path distance.

## Tuning protocol

`scripts/tune_hmm.py` performs a grid search over:

- `transition_sigma_base_m`: 15, 30, 50, 75
- `transition_sigma_fraction`: 0.25, 0.5, 0.75
- `missing_state_penalty`: 3, 6, 9
- `unreachable_penalty`: 8, 12, 16

The script ranks configurations by mean trip-level sequence edit similarity,
then mean arc-set F1. This is a proposed development objective, not an official
metric guarantee. Use only permitted development labels. Never tune against
reserved evaluator labels. Record the predictor mode because fallback scores
are not final ML evidence.

Run a small smoke grid first:

```bash
python scripts/tune_hmm.py --data-root "/path/to/Student folder" --partition student_test --quick --trip-limit 1
```

Run the full grid only after the real Person 1/2 interfaces are connected:

```bash
python scripts/tune_hmm.py --data-root "/path/to/Student folder" --partition student_test
```
