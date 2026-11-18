# Selected PathSnap model

This is the real selected model from the completed `modal-final-20261118` experiment. Training used 80 trips, three grouped folds, six model families and 20 Bayesian trials. The selected tuned histogram gradient boosting model is bundled with its imputation/scaling pipeline.

- `model.joblib`: trusted team-generated trained pipeline.
- `feature_schema.json`: ordered 13-feature schema (version 1.0.0).
- `metadata.json`: model configuration and package versions.
- `training_summary.json`: run provenance, training duration and student-test metrics.

Use the pinned root/ML requirements when loading sklearn artifacts. The backend defaults to this directory. Full experiment logs, fold results and other model checkpoints are in the downloaded `runs/modal-final-20261118/` directory and the Modal Volume. Reference labels are not packaged in this model bundle.
