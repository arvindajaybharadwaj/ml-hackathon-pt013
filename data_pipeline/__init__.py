"""PathSnap Data Pipeline package for Person 1 handoffs.

Exposes the fixed team interface functions:
- load_inputs(root, partition, trip_id) -> TripInputs
- build_candidates(inputs, config) -> CandidateBatch
- load_labels(root, partition, trip_id) -> LabelTable
"""

from .contracts import TripInputs, CandidateBatch, LabelTable
from .loaders import load_inputs, load_labels, load_raw_route
from .candidates import build_candidates
from .alignment import align_route_to_observations
from .features import FEATURE_NAMES

__version__ = "1.0.0"

__all__ = [
    "TripInputs",
    "CandidateBatch",
    "LabelTable",
    "load_inputs",
    "load_labels",
    "load_raw_route",
    "build_candidates",
    "align_route_to_observations",
    "FEATURE_NAMES",
    "__version__",
]
