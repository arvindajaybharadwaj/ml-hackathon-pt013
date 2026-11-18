from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.config import Settings
from backend.app.main import build_services


def settings_from_args(data_root: str | None, partition: str):
    settings = Settings.from_env()
    updates = {"partition": partition}
    if data_root:
        updates["data_root"] = Path(data_root).expanduser().resolve()
    # Experiments may run the explicitly labelled fallback if Person 2's model
    # has not yet been handed over. Results record the predictor mode.
    return replace(settings, **updates)


def create_services(settings):
    return build_services(settings)


def compact(values):
    result = []
    for value in values:
        if value is not None and (not result or result[-1] != int(value)):
            result.append(int(value))
    return result


def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def arc_set_scores(predicted, reference):
    p, r = set(predicted), set(reference)
    overlap = len(p & r)
    precision = overlap / len(p) if p else 0.0
    recall = overlap / len(r) if r else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def sequence_similarity(predicted, reference):
    denom = max(len(predicted), len(reference), 1)
    return 1.0 - levenshtein(predicted, reference) / denom
