# PathSnap ML

See the repository README for installation, real-data training, MPS, Modal, testing, and saved-model evaluation commands.

- `data.py`: reads flat or partitioned CSV/Parquet handoffs, normalises the feature schema, validates identifiers and label separation.
- `models.py`: five sklearn families and Bayesian search spaces.
- `neural.py`: sixth model, a PyTorch MLP supporting MPS/CUDA training and portable CPU inference.
- `train.py`: grouped CV, timing benchmark, optional Bayesian tuning, per-model evidence and final export.
- `predictor.py`: upstream CandidateBatch inference adapter and trusted checkpoint loading.
- `metrics.py`: binary candidate and observation metrics, including missing-prediction errors.
- `evaluate.py`: saved-model student-test evaluation without fitting.
- `modal_train.py`: cloud execution and result download.

The neural model uses binary cross entropy, AdamW, two hidden layers and fixed epochs. Preprocessing fits only training folds. The reported neural loss history is training loss, not validation loss. No comparable loss curves are invented for sklearn models. HMM/Viterbi reconstruction is outside this package.
