"""Baseline candidate and observation metrics. Missing predictions remain errors."""
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, log_loss
from .data import KEY, CANDIDATE_KEY


def evaluate(dataset, scores):
    chosen = scores.sort_values(
        KEY + ['emission_prob', 'candidate_id'], ascending=[True, True, False, True]
    ).drop_duplicates(KEY)
    predictions = dataset.observations[KEY].merge(
        chosen[KEY + ['arc_id']].rename(columns={'arc_id': 'predicted_arc_id'}),
        on=KEY, how='left', validate='one_to_one',
    )
    aligned = dataset.labels[
        dataset.labels.alignment_valid & dataset.labels.reference_arc_id.notna()
    ]
    if aligned.empty:
        raise ValueError('No valid observation-reference labels to evaluate')
    observed = aligned.merge(predictions, on=KEY, validate='one_to_one')
    candidates = dataset.targets().merge(
        scores[CANDIDATE_KEY + ['p_match']], on=CANDIDATE_KEY, validate='one_to_one'
    )
    if candidates.empty:
        raise ValueError('No aligned candidate rows to evaluate')
    y = candidates.y.to_numpy()
    probability = candidates.p_match.to_numpy()
    predicted = probability >= 0.5
    hits = candidates[candidates.y == 1][KEY].drop_duplicates()
    per_trip = []
    for _, trip in observed.groupby('trip_id'):
        actual_arcs = trip.reference_arc_id.astype(int).to_numpy()
        predicted_arcs = trip.predicted_arc_id.fillna(-1).astype(int).to_numpy()
        per_trip.append(float(f1_score(
            actual_arcs, predicted_arcs, average='macro', zero_division=0
        )))
    metrics = {
        'candidate_macro_f1': float(f1_score(y, predicted, labels=[0, 1], average='macro', zero_division=0)),
        'candidate_precision': float(precision_score(y, predicted, zero_division=0)),
        'candidate_recall': float(recall_score(y, predicted, zero_division=0)),
        'candidate_log_loss': float(log_loss(y, probability, labels=[0, 1])),
        'top1_accuracy': float((observed.predicted_arc_id == observed.reference_arc_id).mean()),
        'observation_arc_macro_f1': float(np.mean(per_trip)),
        'candidate_miss_rate': float(1 - len(hits) / len(aligned)),
        'alignment_coverage': float(len(aligned) / len(dataset.observations)),
        'prediction_coverage': float(predictions.predicted_arc_id.notna().mean()),
        'evaluated_observations': len(aligned),
    }
    return metrics, predictions
