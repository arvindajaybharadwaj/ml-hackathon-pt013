> Historical audit before completion. See submission_completion.md for the current resolved items and report deferral.

# PathSnap submission audit — 18 November 2026

## Verdict

The real-data local application and model training work, but the complete hackathon submission is **not ready**. Passing tests and the ten student-test flows do not establish completion of the report, final evaluator deployment or public repository submission.

This was a read-only requirements/code/evidence audit except for creating this audit file. No new training, code fixes, commit or push was performed.

## Authoritative requirements reviewed

All pages of the five supplied PDFs in `PathSnap_student_materials/` were extracted and read:

- Deliverables.pdf: submission checklist and report requirements, especially pages 1–7.
- Models and Training Guidelines.pdf: experimentation, metrics, decoder requirements and final deployment, pages 1–6.
- Main Guidelines.pdf: end-to-end system and two-laptop architecture, pages 1–2.
- Dataset Exploration.pdf: raw data, alignment caveats and leakage rules, pages 1–4.
- Assessment and Plagiarism Policy.pdf: reproducibility, truthful evidence and team understanding, pages 1–3.

## Requirement checklist

| Requirement | Status | Verified evidence / remaining work |
|---|---|---|
| Raw data inspection and feature engineering source | Implemented, documentation needs correction | data_pipeline loaders/candidates/features/alignment/export and evidence/data outputs exist. Several numerical claims in the notes disagree with actual exported labels. |
| At least five distinct models | Complete locally | Six trained families in runs/modal-final-20261118/comparison.csv. |
| Training on 80 trips; comparison on 10 student-test trips | Complete locally | Completed run manifest records the full split, not smoke trip limits. |
| Group-aware validation and training-only fitted preprocessing | Implemented and tested | Three whole-trip folds; sklearn pipelines fit imputation/scaling within folds. |
| Bayesian search and tuned/base comparison | Complete locally | Twenty Optuna trials, SQLite study, trial CSV, base/tuned rows and saved configurations. |
| Trained model and preprocessing artifacts | Complete locally, not submitted | artifacts/model contains the real selected bundle, schema, metadata and training summary. This directory is untracked in Git. |
| Required model-comparison plots | Missing | EDA PNGs exist, but no full-run CV F1, student-test F1/Top-1 or log-loss comparison figures exist. |
| Explanation for every experimented model | Missing as a final deliverable | Final report must discuss all six families with actual settings, objectives, results and overfit/underfit evidence; generic algorithm descriptions are insufficient. |
| HMM/Viterbi inference | Implemented and tested locally | Real selected model scores feed variable candidate states, log-space dynamic programming and directed-network transitions. |
| Classifier/HMM route comparison | Partially complete | All ten real student-test flows have sequence, set-F1 and continuity evidence in evidence/integration/api_checks.json. Aligned observation-level HMM Top-1/F1 are not evaluated by the backend. |
| Decoder parameter optimisation | Not complete for current trained model | backend/results/hmm_experiments_best.json is only an eight-configuration, one-trip development_fallback experiment. Current production uses default transition settings. Rerun a justified development search with the actual trained model and save/load the chosen config. |
| Folium visualisation | Partially complete | Live map has GPS, nearby roads, classifier choices and HMM matched segments with controls. A report/offline map including the available student-data reference route is missing. Live evaluator UI must continue to exclude reference labels. |
| API and UI communicate using real model | Complete locally | Live dashboard and all ten student-test trips were verified through the Vite proxy; no invented trip IDs or distance fallback in production defaults. |
| Final two-laptop demonstration | Not verified | Only a same-machine deployment was tested. Need a real Laptop 1 API / Laptop 2 UI LAN run, and capture receiving API request logs. |
| Ten reserved evaluator trips | Not yet validated / external inputs required | Only 80 train and 10 student-test trips exist locally. Need authorised evaluator inputs and candidate generation/configuration, then check all ten without using their labels for selection or tuning. |
| One consolidated technical PDF report | Missing | No final report PDF exists. PT-013_Team_Execution_Guide.pdf is an execution guide, not the required final methodology/results report. |
| Report map/UI/API screenshots | Missing | No committed screenshots of the completed real-model UI, report map and receiving backend terminal were found. Earlier fallback text smoke logs are not substitutes. |
| Public GitHub repository | Public, submission incomplete | GitHub API confirmed https://github.com/arvindajaybharadwaj/ml-hackathon-pt013 is PUBLIC with main as default. Current integration/training changes and selected artifact remain modified/untracked locally. |
| Experiment records available to evaluators | Not submitted | Full comparison, folds, configs, tuning CSV/database and metrics are inside gitignored runs/. Copy required lightweight evidence into a tracked evidence directory before committing. |
| README/install/start instructions | Implemented locally, not submitted | Root/backend/frontend READMEs and scripts exist, but current edits are uncommitted. |
| Reproducible raw-export command | Needs correction | data_pipeline/export.py still hardcodes Windows F:/... paths in its __main__. Export function itself exists, but the direct CLI is not portable. |
| Team-authored interpretation / viva / AI disclosure | Cannot be verified by code | Team must check claims, explain choices/results and follow the event's disclosure rules. |

## Training outcome verified

Source: completed run manifest, comparison.csv, selected metadata and independently re-evaluated metrics.

- Training duration: 866.08 seconds (14.43 minutes).
- Training candidates: 1,474,434; student-test candidates: 164,457; 13 features.
- Selected model: tuned histogram gradient boosting.
- Base/tuned CV candidate Macro F1: 0.880544 / 0.880768.
- Selected student-test candidate Macro F1: 0.894405.
- Selected observation Top-1: 0.823859, evaluated on 17,821 aligned observations of 19,160 total.
- Selected observation arc Macro F1: 0.662506.
- Student-test alignment coverage: 0.930115.
- Classifier selection/tuning used training CV rather than student-test scores.

These metrics do not establish HMM observation-level accuracy. Coverage and the label alignment procedure must be described alongside accuracy.

## Data-report discrepancies found by recomputation

Read alignment_valid directly from each exported labels.parquet:

| Partition | Actual coverage | Existing EDA handoff claim |
|---|---:|---:|
| train | 85.0962% | 93.4% |
| student_test | 93.0115% | 99.8% |

The handoff notes also claim 97.12% of GPS steps are exactly one second. Recomputing np.isclose(delta_t, 1) from the raw tracks gives 201,053 / 208,337 training intervals (96.5037%) and 18,886 / 19,150 student-test intervals (98.6214%), or about 96.68% combined. The exporter plot label hardcodes a different figure. Correct notes/plot annotations before using them in the report.

The claim of <0.01% projection distortion over 50 km is written for Darmstadt, while these trips cover widely separated regions. It needs a justified measurement or a qualified explanation; this audit does not validate that bound. The feature schema's documented value ranges and first-observation policies should also be reconciled with the real computed values before presenting them as validated guarantees.

## Implementation issues to address before claiming robust route completion

1. HMM pointwise evaluation is missing. backend EvaluationService returns sequence/set metrics against .route, while aligned GPS labels are available separately. Add server-side aligned classifier/HMM Top-1 and F1 with explicit coverage and missing-prediction treatment.
2. Continuity hides missing endpoints. _route_continuity removes pairs with a None endpoint and returns 1.0 if no valid pair remains. All ten current student-test trips happened to have candidates, but this overstates continuity on missing-candidate cases. Define the denominator over original adjacent observations and test gaps.
3. Returned geometry is not a complete traversed path. PredictionService returns endpoints of selected observation arcs and does not insert intermediate directed-network edges. route_coordinates can concatenate disconnected edges. The Folium map avoids drawing invented connector lines by rendering segments individually. Decide whether to expand valid shortest paths and represent breaks explicitly; describe the actual output honestly in the report.
4. Some experiment scripts accept an arbitrary --partition for HMM tuning. Enforce an explicit train/student_test allowlist before final evaluator data is configured, so accidental evaluator tuning is rejected.
5. Reference map requirement and inference isolation must be satisfied separately: create an offline/report map using permitted student data with its reference layer, while keeping the live evaluator UI reference-free.

These findings do not negate successful local API/UI execution; they limit claims of final evaluation completeness and robustness.

## Suggested completion order

1. Correct numerical EDA claims and the raw-export CLI; address missing-observation continuity and aligned HMM metrics.
2. Run documented HMM tuning using the real selected model on permitted development data; persist the chosen configuration and rerun the ten-trip comparison.
3. Produce the required model plots, real neural training-history plot (do not invent an unrecorded validation history), and offline Folium reference comparison map. Add UI/map/API terminal screenshots.
4. Prepare one consolidated PDF with the mandated sections and model-specific evidence/analysis; review it with the team.
5. Copy lightweight full-run evidence out of ignored runs/, stage the real model and all required source, inspect staged content for secrets, then commit and push when authorised.
6. Rehearse on two actual laptops; when organisers provide reserved trips, validate the full inference/evaluation flow for all ten without retuning on evaluator outcomes.

Do not label the submission complete until these remaining requirements are satisfied or an organiser explicitly exempts an item.
