from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
import pandas as pd


class CandidatePredictor(Protocol):
    name: str
    mode: str

    def score_candidates(self, candidates: pd.DataFrame) -> np.ndarray: ...


class BundleCandidatePredictor:
    """Use the versioned ML bundle and its training-time feature order unchanged."""
    mode = 'trained_model'

    def __init__(self, directory: Path):
        from ml.predictor import load_model
        self.predictor = load_model(directory)
        self.name = self.predictor.metadata['model_name']

    def score_candidates(self, candidates: pd.DataFrame) -> np.ndarray:
        scores = self.predictor.score_candidates({
            'candidates': candidates,
            'schema_version': candidates.attrs.get('schema_version'),
        })
        return scores.p_match.to_numpy(dtype=float)


class DistanceFallbackPredictor:
    """Development-only scoring fallback. It is NOT Person 2's trained ML model."""
    name = "distance-only-development-fallback"
    mode = "development_fallback"

    def __init__(self, sigma_m: float = 25.0):
        self.sigma_m = float(sigma_m)

    def score_candidates(self, candidates: pd.DataFrame) -> np.ndarray:
        distance = candidates.distance_m.to_numpy(dtype=float)
        return np.exp(-0.5 * np.square(distance / max(self.sigma_m, 1e-6)))


class JoblibCandidatePredictor:
    """Adapter for Person 2's saved sklearn/joblib model.

    `binary` mode expects one row per candidate and a positive class (normally 1).
    `arc_multiclass` mode expects the classifier's class labels to be road arc IDs.
    `direct` mode uses a one-column probability or a decision/prediction score.
    """

    name = "person-2-joblib-model"

    def __init__(self, model_path: Path, feature_columns: tuple[str, ...], mode: str = "binary", positive_class: str = "1"):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model artifact not found: {self.model_path}")
        self.model = joblib.load(self.model_path)
        self.feature_columns = feature_columns
        self.mode = mode
        self.positive_class = positive_class
        if not self.feature_columns:
            feature_path = self.model_path.with_name("model_features.json")
            if feature_path.is_file():
                self.feature_columns = tuple(json.loads(feature_path.read_text()))
        if not self.feature_columns:
            raise ValueError("Set PATHSNAP_MODEL_FEATURES or provide model_features.json")

    def score_candidates(self, candidates: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.feature_columns if c not in candidates.columns]
        if missing:
            raise ValueError(f"Person 2 model requires missing feature columns: {missing}")
        X = candidates.loc[:, self.feature_columns]
        if hasattr(self.model, "predict_proba"):
            probabilities = np.asarray(self.model.predict_proba(X), dtype=float)
            classes = list(getattr(self.model, "classes_", []))
            if self.mode == "binary":
                index = _find_class_index(classes, self.positive_class)
                if probabilities.ndim != 2 or index is None:
                    raise ValueError("Cannot identify the positive class in Person 2 model")
                scores = probabilities[:, index]
            elif self.mode == "arc_multiclass":
                if probabilities.ndim != 2 or not classes:
                    raise ValueError("Expected a multiclass predict_proba model")
                scores = np.zeros(len(candidates), dtype=float)
                for i, arc_id in enumerate(candidates.arc_id.astype(int)):
                    class_index = _find_class_index(classes, str(arc_id))
                    if class_index is None:
                        class_index = _find_class_index(classes, arc_id)
                    if class_index is not None:
                        scores[i] = probabilities[i, class_index]
            else:
                if probabilities.ndim == 2 and probabilities.shape[1] == 1:
                    scores = probabilities[:, 0]
                elif probabilities.ndim == 2 and probabilities.shape[1] == 2:
                    scores = probabilities[:, 1]
                else:
                    raise ValueError("Use binary or arc_multiclass mode for multi-column probabilities")
        elif hasattr(self.model, "decision_function"):
            raw = np.asarray(self.model.decision_function(X), dtype=float).reshape(-1)
            scores = 1.0 / (1.0 + np.exp(-np.clip(raw, -40, 40)))
        else:
            raw = np.asarray(self.model.predict(X), dtype=float).reshape(-1)
            scores = 1.0 / (1.0 + np.exp(-np.clip(raw, -40, 40)))
        if len(scores) != len(candidates):
            raise ValueError("Predictor returned a different number of scores than candidates")
        return np.clip(np.nan_to_num(scores, nan=1e-9, posinf=1.0, neginf=1e-9), 1e-9, 1.0)


def _find_class_index(classes, target):
    target_text = str(target)
    for i, value in enumerate(classes):
        if str(value) == target_text:
            return i
    return None
