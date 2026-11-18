# EDA & Data Pipeline Notes for Report Author (Person 4)

**From:** Person 1 (Data Engineer)  
**To:** Person 4 (Frontend and Report Engineer)  
**Deliverable Context:** Supporting text, tables, verified figures, and answers for **Section 2.2: Dataset Understanding and Exploratory Data Analysis** of the final technical report.

---

## 1. Direct Answers to Mandatory Section 2.2 Questions

### Question 1: How did you preprocess the data?
* **Partition Used:** All 80 training trips (`train/`) and 10 student-testing trips (`student_test/`). Evaluator data was kept strictly untouched.
* **Coordinate Projection:** Raw coordinates are angular WGS84 coordinates ($\text{lon}, \text{lat}$). We implemented a localized Equirectangular metric projection centered at each trip's observation centroid $(\bar{\lambda}, \bar{\phi})$:
  $$\Delta x = (\lambda - \bar{\lambda}) \times 111320.0 \times \cos(\bar{\phi}_{\text{rad}})$$
  $$\Delta y = (\phi - \bar{\phi}) \times 111320.0$$
  This yields metric Cartesian coordinates where $\Delta x$ is Easting and $\Delta y$ is Northing in metres. In Darmstadt, Germany (~$49.85^\circ\text{N}, 8.65^\circ\text{E}$), this introduces less than $0.01\%$ distortion over a 50 km trajectory extent while preserving high computational speed.
* **Spatial Candidate Generation (Reference-Free):** Using `shapely.STRtree` (GEOS R-Tree), directed road segments (arcs) within a $50.0$-metre search radius of each GPS point are identified. For each candidate segment, the orthogonal/clamped projection point, snapped WGS84 coordinates, and projection fraction $t_{\text{proj}} \in [0, 1]$ are computed. At most 15 closest candidate segments are retained per observation. Observations with zero candidates within 50m (<0.05%) are strictly retained in the observations table to prevent evaluation denominator skew.
* **Separation of Inputs and Reference Labels:** The input pipeline (`load_inputs` and `build_candidates`) operates completely blind to the `.route` file. Ground-truth route labels are processed independently in `load_labels`.
* **Missing-Value Policy:** No missing values exist in raw files. Computed candidate features are fully imputed with physical bounding defaults (e.g. neutral heading difference $90^\circ$ and cosine alignment $0.0$ when stationary). Parquet tables contain zero NaNs.

---

### Question 2: What patterns did you observe during data exploration? What features did you extract and why?

#### Observed Exploration Patterns:
1. **Sampling Frequency Dynamics:** 97.12% of consecutive GPS observations have a sampling interval of exactly $\Delta t = 1.0$ s (nominal 1 Hz). However, 23 of 90 trips contain large time gaps exceeding 60 seconds (up to 42.5 hours in trip `00000045`), indicating prolonged halts, parking, or tracking reactivation.
2. **Stationary / Idling Clusters:** 18.4% of GPS observations exhibit speeds below $0.5$ m/s (~1.8 km/h). During halts (traffic signals, parking), GPS multipath and sensor noise cause artificial wandering points that lack meaningful heading vectors.
3. **Non-1-to-1 Relationship of Routes to Observations:** The `.route` file contains an ordered sequence of traversed arc IDs, but vehicle dwell time on individual segments causes multiple GPS points to map to a single arc, while fast driving across short segments skips intermediate observations.
4. **Repeated Traversed Segments:** 10 trips contain repeated road segments (the vehicle travels along the same road segment at two distinct stages in the trip). Naive nearest-arc matching fails on repeated segments; an order-aware chronological forward alignment is mandatory.
5. **Partial Ground-Truth Truncation:** In trip `00000054`, the `.route` reference terminates after arc 421 (~2,268 seconds) while GPS tracking continues for 19,313 seconds. Our alignment explicitly identifies this as `route_extent_exceeded`, preventing false training signal.

#### Extracted Features & Rationale:
The team extracted 13 distinct candidate features (defined in `feature_schema.json`):

| # | Feature Name | Unit | Physical & Modeling Rationale |
| :--- | :--- | :--- | :--- |
| 1 | `distance_to_segment_m` | m | Strongest baseline indicator of match likelihood under Gaussian GPS noise models. |
| 2 | `projection_fraction` | $[0, 1]$ | Quantifies vehicle progress along the directed arc ($0 = \text{start}$, $1 = \text{end}$). |
| 3 | `segment_length_m` | m | Road topology scale; distinguishes short intersection connectors from arterial roads. |
| 4 | `heading_diff_deg` | $[0, 180]^\circ$ | Angular difference between trajectory vector and arc vector; penalizes opposing orientations. |
| 5 | `heading_alignment_cos` | $[-1, 1]$ | Continuous cosine alignment; $+1$ indicates identical flow, $-1$ indicates counterflow/one-way violation. |
| 6 | `gps_speed_mps` | m/s | Empirical speed; differentiates walking/idling from high-speed motorway transit. |
| 7 | `time_gap_s` | s | Interval $\Delta t$; detects tracking dropouts and non-1Hz irregularities. |
| 8 | `spatial_gap_m` | m | Step displacement; detects GPS teleportation/multipath anomalies. |
| 9 | `candidate_rank_dist` | $0 \dots 14$ | Relative proximity rank; normalizes distance across dense urban street grids. |
| 10 | `dist_diff_from_nearest_m` | m | Distance margin above nearest candidate; sharpens margin for ambiguous parallel streets. |
| 11 | `speed_length_ratio` | unitless | Transition indicator; measures if current speed is sufficient to traverse the segment. |
| 12 | `is_zero_speed` | $\{0, 1\}$ | Disables noisy heading penalties when the vehicle is stationary. |
| 13 | `reliability_flag` | $\{0, 1\}$ | Quality filter ensuring regular 1Hz sampling, plausible speed ($\leq 45$ m/s), and proximity. |

---

## 2. Dataset Summary Table (For Report Embedding)

| Partition | Trips | Observations | Mean Obs/Trip | Total Candidates | Mean Candidates/Obs | Valid Alignment Coverage |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Training (`train`)** | 80 | 208,417 | 2,605.2 | 1,474,434 | 7.07 | 93.4% (excl. 00000054: 98.9%) |
| **Student Testing (`student_test`)** | 10 | 19,160 | 1,916.0 | 164,457 | 8.58 | 99.8% |
| **Combined Total** | **90** | **227,577** | **2,528.6** | **1,638,891** | **7.20** | **93.9%** |

*Note: Training partition coverage includes trip `00000054` where the GPS track continues for 17,000 points after the route finishes. Excluding this known anomaly, training alignment coverage is 98.9%.*

---

## 3. Evidence Figures for Final Report

The following high-resolution figures have been produced and saved in `evidence/data/` for inclusion in the PDF report:

1. `sampling_interval_dist.png`: Histogram showing GPS sampling intervals ($\Delta t \leq 5$s) highlighting the 97.1% 1 Hz peak.
2. `speed_distribution.png`: Histogram of vehicle speeds across all observations with urban (50 km/h) and highway (120 km/h) markers.
3. `candidate_count_dist.png`: Distribution of road segment candidates per GPS observation under $R \leq 50$m (mean: 7.20).
4. `candidate_distance_dist.png`: Distribution of candidate Euclidean distances to GPS observations.
5. `alignment_coverage.png`: Bar chart of route alignment diagnostic breakdown (`valid_aligned`, `route_extent_exceeded`, `distance_exceeds_threshold`).
6. `network_scale_summary.png`: Boxplots of road network node and arc counts, alongside ground-truth route arc lengths.
