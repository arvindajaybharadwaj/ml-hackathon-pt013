# PathSnap Data Engineering & Dataset Exploration Notes

**Author:** Person 1 (Data Engineer)  
**Package:** `data_pipeline`  
**Date:** 18 November 2026  
**Scope:** Dataset inspection, raw loaders, spatial candidate generation, feature engineering, and order-aware reference route alignment.

---

## 1. Executive Summary & Inventory Confirmation

An exhaustive inspection of the supplied raw dataset in `Student folder/` was conducted. All 90 supplied trips were systematically loaded, validated, and processed:
- **Total Trips:** 90
- **Training Partition (`train`):** 80 trips (IDs `00000001` through `00000099`)
- **Student Testing Partition (`student_test`):** 10 trips (`00000000`, `00000007`, `00000009`, `00000040`, `00000052`, `00000056`, `00000065`, `00000076`, `00000077`, `00000091`)
- **Evaluator Testing Partition:** 10 reserved trips (kept untouched and isolated for final evaluation).
- **Total GPS Observations:** 227,577 observations (208,417 train; 19,160 student_test).
- **Total Road Candidates Generated ($R \leq 50$m, max 15):** 1,638,891 candidates (1,474,434 train; 164,457 student_test).
- **Mean Candidates per Observation:** 7.20 across all trips (median: 7.0, 95th percentile: 14.0).
- **Zero-Candidate Rate:** < 0.05% of observations had zero road segments within 50m. All zero-candidate observations are strictly preserved in `observations.parquet` to ensure unbiased denominator counts in downstream Top-1 evaluation.

---

## 2. Raw File Specifications & Semantics

Each trip directory `partition/trip_id/` contains four raw whitespace-separated text files without headers:

| File Extension | Content Description | Column Specification | Indexing Semantics |
| :--- | :--- | :--- | :--- |
| `.track` | Raw GPS trajectory observations | `col0: lon` (float64), `col1: lat` (float64), `col2: timestamp_s` (float64) | 0-based contiguous row index assigned as `obs_id` |
| `.nodes` | Geographical junction nodes of road network | `col0: lon` (float64), `col1: lat` (float64) | 0-based contiguous row index assigned as `node_id` |
| `.arcs` | Directed road network segments | `col0: from_node_id` (int64), `col1: to_node_id` (int64) | 0-based contiguous row index assigned as `arc_id` |
| `.route` | Chronological sequence of ground-truth arcs | `col0: arc_id` (int64) traversed by the vehicle | 0-based chronological route sequence index |

### Critical Findings & Validations:
1. **Coordinate Order:** Longitude is strictly column 0, and Latitude is column 1 in both `.track` and `.nodes`.
2. **Local ID Scopes:** Node IDs and Arc IDs are strictly local to each individual trip's road network. They must never be merged into a global categorical class in machine learning models.
3. **Data Completeness:** 100% of the 90 trips contain all four files. No missing files, truncated lines, or NaN/Inf coordinate values exist in the raw dataset.
4. **Node Reference Integrity:** Every `from_node_id` and `to_node_id` in `.arcs` references a valid node within `[0, len(nodes)-1]`.
5. **Route Arc Integrity:** Every `arc_id` in `.route` references a valid arc within `[0, len(arcs)-1]`.

---

## 3. Coordinate System & Geometric Projection Formulation

The raw coordinates are provided as angular WGS84 coordinates ($\text{EPSG:4326}$) centered in the region of Darmstadt / Hesse, Germany:
- **Longitude Range:** $8.50^\circ\text{E} \text{ to } 8.85^\circ\text{E}$
- **Latitude Range:** $49.75^\circ\text{N} \text{ to } 50.05^\circ\text{N}$

### Justification of Local Metric Projection:
Direct Euclidean distance in degrees $(\Delta\text{lon}^2 + \Delta\text{lat}^2)^{1/2}$ produces distorted distances because 1 degree of longitude at $49.85^\circ\text{N}$ is substantially shorter than 1 degree of latitude ($\cos(49.85^\circ) \approx 0.6448$).

To achieve millimetre-level geometric precision with $O(1)$ scalar computational efficiency, we employ a localized metric Equirectangular projection centered on each trip's observation centroid $(\bar{\lambda}, \bar{\phi})$:
$$\Delta x = (\lambda - \bar{\lambda}) \times 111320.0 \times \cos(\bar{\phi}_{\text{rad}})$$
$$\Delta y = (\phi - \bar{\phi}) \times 111320.0$$

Where $x$ represents Easting in metres and $y$ represents Northing in metres. Over a 50 km local extent, distortion is strictly below 0.01% compared to ellipsoidal geodesic Vincenty calculations, while enabling vectorized GEOS and NumPy operations.

---

## 4. Temporal Dynamics & Data Quality Anomalies

### 4.1 Sampling Interval Distribution ($\Delta t$)
- **Nominal Sampling Rate:** 97.12% of consecutive GPS observations have $\Delta t = 1.00$ s exactly.
- **Median Interval:** 1.00 s.
- **Micro-intervals:** < 0.1% of observations exhibit $\Delta t < 0.5$ s due to GPS receiver jitter.

### 4.2 Anomalous Temporal Gaps
- 23 out of 90 trips contain at least one gap $\Delta t > 60$ s (e.g. vehicle parked, tunnel dropout, or tracking resumed after an extended halt).
- **Maximum Observed Gap:** 153,064 seconds (~42.5 hours) in trip `00000045`.
- **Pipeline Handling:** Observations following large gaps are explicitly flagged in `reliability_flag = 0` and kinematic features forward-fill or clamp $\Delta t$ to prevent numerical instability. In route alignment, large gaps automatically expand the forward search window dynamically.

### 4.3 Kinematic Speed Observations
- Speeds are computed as step Euclidean distance divided by $\Delta t$.
- Normal vehicle speeds range from 0 km/h (idling at traffic lights) to ~130 km/h on regional motorways (Autobahn).
- Stationary periods (speed $< 0.5$ m/s, or 1.8 km/h) account for 18.4% of observations, flagged via `is_zero_speed = 1`. During stationary periods, trajectory heading is physically indeterminate; heading difference is defensively set to $90.0^\circ$ (neutral) with cosine alignment $0.0$.

---

## 5. Candidate Generation Methodology (Reference-Free)

Road segment candidate generation is executed strictly reference-free:
1. **Spatial Index:** Directed road segments are indexed using `shapely.STRtree` (a C-implemented Sort-Tile-Recursive R-Tree via GEOS).
2. **Search Radius:** Search radius $R = 50.0$ metres.
3. **Point-to-Segment Projection:**
   For GPS point $p = (p_x, p_y)$ and segment from $u = (a_x, a_y)$ to $v = (b_x, b_y)$ with displacement vector $d = v - u$:
   $$t_{\text{proj}} = \text{clamp}\left( \frac{(p - u) \cdot d}{\|d\|^2}, 0.0, 1.0 \right)$$
   $$\text{snap}(p) = u + t_{\text{proj}} \cdot d$$
   $$\text{dist}(p, S) = \|p - \text{snap}(p)\|_2$$
4. **Snapped WGS84 Inversion:** Snapped metric coordinates are inverted back to WGS84 $(\text{snapped\_lon}, \text{snapped\_lat})$ with full double precision.
5. **Pruning & Zero-Candidate Guarantee:** Candidates are ranked by distance ascending; at most 15 nearest candidates are retained per observation. Observations with zero candidates are preserved in the `observations` table with empty candidate slices.

---

## 6. Feature Schema & Engineering Rationale

All 13 features are documented in `feature_schema.json` and computed in exact fixed order:

1. `distance_to_segment_m` ($\text{float32}$, m): Primary spatial proximity indicator.
2. `projection_fraction` ($\text{float32}$, $[0, 1]$): Positional state along directed arc ($0 = \text{from\_node}$, $1 = \text{to\_node}$).
3. `segment_length_m` ($\text{float32}$, m): Physical arc length in metres.
4. `heading_diff_deg` ($\text{float32}$, $[0, 180]^\circ$): Trajectory bearing vs arc bearing angular divergence.
5. `heading_alignment_cos` ($\text{float32}$, $[-1, 1]$): Directional alignment metric ($+1$ forward, $-1$ counterflow).
6. `gps_speed_mps` ($\text{float32}$, m/s): Vehicle velocity magnitude.
7. `time_gap_s` ($\text{float32}$, s): Elapsed interval $\Delta t$.
8. `spatial_gap_m` ($\text{float32}$, m): Step displacement.
9. `candidate_rank_dist` ($\text{int32}$, $0 \dots 14$): Distance rank among competing candidates.
10. `dist_diff_from_nearest_m` ($\text{float32}$, m): Margin above nearest candidate ($0.0$ for rank 0).
11. `speed_length_ratio` ($\text{float32}$): Step travel distance relative to segment length.
12. `is_zero_speed` ($\text{int32}$, $\{0, 1\}$): Stationary vehicle flag.
13. `reliability_flag` ($\text{int32}$, $\{0, 1\}$): Quality flag combining sampling regularity, plausible speed, and proximity.

**Missing-Value Policy:** 100% deterministic imputation. Zero NaNs or nulls in candidate feature matrices.

---

## 7. Order-Aware Route Alignment & Ground-Truth Diagnostics

### 7.1 Why Simple Nearest-Arc Alignment Fails
Supplied `.route` files contain the sequence of arcs traversed by the vehicle, but **do not correspond 1-to-1 with GPS rows**:
- A vehicle traveling slowly or waiting at an intersection generates multiple consecutive GPS points on the same arc.
- A vehicle traveling fast covers multiple short arcs in 1 second.
- 10 trips contain **repeated road segments** (the vehicle traversed the same arc at two different times in the trip). A simple nearest-segment lookup confuses the first pass with the second pass.

### 7.2 Order-Aware Alignment Algorithm
We implement a forward-progress tracking window along the chronological route indices $r \in [0, M-1]$:
- An active route index pointer $r_{\text{curr}}$ begins at the route start.
- At observation $t$, candidates are evaluated within a forward-biased window $[r_{\text{curr}} - 1, r_{\text{curr}} + W]$.
- If a route segment lies within $35.0$ metres, the observation is marked `alignment_valid = True`, `alignment_reason = 'valid_aligned'`, and $r_{\text{curr}}$ advances.
- If distance exceeds $35.0$ metres:
  - If $r_{\text{curr}} \geq M - 2$, the trajectory has continued beyond the recorded route extent (`alignment_reason = 'route_extent_exceeded'`).
  - If a large time gap preceded the point, it is flagged as `temporal_gap_break`.
  - Otherwise flagged as `distance_exceeds_threshold`.
  - In all invalid cases, `reference_arc_id` is recorded as `pd.NA` (nullable integer).

### 7.3 Ground-Truth Coverage Results:
- **Typical Trips (e.g. 00000001, 00000000, 00000007):** Valid coverage is 99.8% to 100.0%.
- **Special Cases Documented:**
  - `00000054`: GPS tracking continued for 19,313 points while the `.route` ended after arc 421 (~point 2,268). Valid route coverage is 11.7%, with 86.0% correctly identified as `route_extent_exceeded`.
  - `00000002`: Vehicle stopped/diverged, valid coverage 86.7%.
- Downstream models must evaluate Top-1 accuracy only over valid ground-truth observations (`alignment_valid == True`) or treat candidate misses as explicit errors.

---

## 8. Verification & Delivery Manifest

The following deliverables are prepared in `F:\ML_QUIZ\New folder\`:

1. `data_pipeline/`:
   - `__init__.py`: Package entry point exposing `load_inputs`, `build_candidates`, `load_labels`.
   - `contracts.py`: Dataclass models `TripInputs`, `CandidateBatch`, `LabelTable`.
   - `loaders.py`: Raw loaders strictly separating inputs from labels.
   - `candidates.py`: Spatial indexing and reference-free candidate generation.
   - `features.py`: Feature calculation and schema enforcement.
   - `alignment.py`: Order-aware route alignment engine.
   - `export.py`: Batch partition processor.
2. `prepared_data/`:
   - `train/observations.parquet`, `train/candidates.parquet`, `train/labels.parquet`
   - `student_test/observations.parquet`, `student_test/candidates.parquet`, `student_test/labels.parquet`
   - `pipeline_config.json`: Configuration and dataset metadata.
3. `evidence/data/`:
   - `inventory.csv`: Full 90-trip metadata table.
   - `data_notes.md`: Detailed engineering report.
   - `eda_notes_for_person4.md`: Synthesis for final technical report Section 2.2.
   - Diagnostic figures: `sampling_interval_dist.png`, `speed_distribution.png`, `candidate_count_dist.png`, `candidate_distance_dist.png`, `alignment_coverage.png`, `network_scale_summary.png`.
4. `feature_schema.json`: Complete specification of feature order, units, types, and imputation.
5. `requirements_data.txt`: Clean pip-compatible dependencies.
6. `example_usage.py`: Runnable handoff verification script.
7. `tests/data_pipeline/`: 12 automated pytest test cases verifying loaders, candidates, features, alignment, and leakage prevention.
