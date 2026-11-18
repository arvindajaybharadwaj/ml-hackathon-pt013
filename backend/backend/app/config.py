from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Runtime configuration. Values can be set with environment variables."""

    data_root: Path
    partition: str
    candidate_dir: Path | None
    model_path: Path | None
    model_features: tuple[str, ...]
    model_mode: str
    positive_class: str
    allow_dev_fallback: bool
    candidate_radius_m: float
    max_candidates: int
    emission_sigma_m: float
    transition_sigma_base_m: float
    transition_sigma_fraction: float
    missing_state_penalty: float
    unreachable_penalty: float
    cors_origins: tuple[str, ...]

    @classmethod
    def from_env(cls) -> "Settings":
        data_root = Path(os.getenv("PATHSNAP_DATA_ROOT", "./data")).expanduser().resolve()
        candidate_dir_raw = os.getenv("PATHSNAP_CANDIDATE_DIR", "").strip()
        model_path_raw = os.getenv("PATHSNAP_MODEL_PATH", "").strip()
        features_raw = os.getenv("PATHSNAP_MODEL_FEATURES", "").strip()
        origins_raw = os.getenv(
            "PATHSNAP_CORS_ORIGINS",
            "http://localhost:3000,http://localhost:5173,http://127.0.0.1:5173",
        )
        return cls(
            data_root=data_root,
            partition=os.getenv("PATHSNAP_PARTITION", "student_test").strip(),
            candidate_dir=Path(candidate_dir_raw).expanduser().resolve() if candidate_dir_raw else None,
            model_path=Path(model_path_raw).expanduser().resolve() if model_path_raw else None,
            model_features=tuple(x.strip() for x in features_raw.split(",") if x.strip()),
            model_mode=os.getenv("PATHSNAP_MODEL_MODE", "distance_fallback").strip().lower(),
            positive_class=os.getenv("PATHSNAP_POSITIVE_CLASS", "1").strip(),
            allow_dev_fallback=_as_bool(os.getenv("PATHSNAP_ALLOW_DEV_FALLBACK"), True),
            candidate_radius_m=float(os.getenv("PATHSNAP_CANDIDATE_RADIUS_M", "100")),
            max_candidates=int(os.getenv("PATHSNAP_MAX_CANDIDATES", "8")),
            emission_sigma_m=float(os.getenv("PATHSNAP_EMISSION_SIGMA_M", "25")),
            transition_sigma_base_m=float(os.getenv("PATHSNAP_TRANSITION_SIGMA_BASE_M", "30")),
            transition_sigma_fraction=float(os.getenv("PATHSNAP_TRANSITION_SIGMA_FRACTION", "0.5")),
            missing_state_penalty=float(os.getenv("PATHSNAP_MISSING_STATE_PENALTY", "6")),
            unreachable_penalty=float(os.getenv("PATHSNAP_UNREACHABLE_PENALTY", "12")),
            cors_origins=tuple(x.strip() for x in origins_raw.split(",") if x.strip()),
        )
