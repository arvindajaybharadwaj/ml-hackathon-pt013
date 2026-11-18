"""Check real model → API → frontend proxy for all available trips."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:5173')
    parser.add_argument('--output', type=Path, default=Path('evidence/integration/api_checks.json'))
    args = parser.parse_args()
    rows = []
    with httpx.Client(base_url=args.base_url, timeout=180) as client:
        def get(path):
            response = client.get(path)
            response.raise_for_status()
            return response.json()
        health = get('/health')
        assert health['model_loaded'] and health['prediction_mode'] == 'trained_model', health
        trips = get('/api/trips')['trip_ids']
        for trip_id in trips:
            started = perf_counter()
            response = client.post(f'/api/predict/{trip_id}')
            response.raise_for_status()
            prediction = response.json()
            assert prediction['gps_observations'] == len(prediction['predicted_arc_ids'])
            assert 'reference_arc_id' not in response.text
            evaluation = get(f'/api/evaluate/{trip_id}')
            route_map = client.get(f'/api/maps/{trip_id}')
            route_map.raise_for_status()
            assert 'HMM matched roads' in route_map.text
            rows.append({'trip_id': trip_id, 'observations': prediction['gps_observations'],
                         'candidate_rows': prediction['candidate_rows'],
                         'matched_observations': prediction['matched_observations'],
                         'hmm_continuity': prediction['route_continuity'],
                         'classifier_continuity': prediction['classifier_only_route_continuity'],
                         'sequence_similarity': evaluation['sequence_edit_similarity'],
                         'classifier_sequence_similarity': evaluation['classifier_only_sequence_edit_similarity'],
                         'route_arc_set_f1': evaluation['arc_set_f1'],
                         'wall_seconds': round(perf_counter()-started, 3)})
            # Save each success so interrupted checks remain reviewable.
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({'health': health, 'checked': rows,
                                              'complete': len(rows) == len(trips)}, indent=2)+'\n')
            print(f'{trip_id}: prediction, evaluation and map OK ({rows[-1]["wall_seconds"]}s)', flush=True)
    print(f'All {len(rows)} trips verified through {args.base_url}')


if __name__ == '__main__':
    main()
