"""Prepare inference candidates for a supplied partition without reading reference routes."""
import argparse
import json
from pathlib import Path
import shutil
import pandas as pd
from data_pipeline import load_inputs, build_candidates


def prepare(raw_root, partition, output):
    if not partition or Path(partition).name != partition or partition in {'.','..'}:
        raise ValueError('Partition must be a single directory name')
    directory = raw_root / partition
    ids = sorted(p.name for p in directory.iterdir() if p.is_dir() and len(p.name)==8 and p.name.isdigit())
    if not ids:
        raise ValueError('No eight-digit trip folders found')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose an empty output directory')
    observations, candidates = [], []
    for tid in ids:
        inputs = load_inputs(raw_root, partition, tid)
        batch = build_candidates(inputs)
        observations.append(batch.observations.assign(trip_id=tid, partition=partition))
        candidates.append(batch.candidates)
        print(f'{tid}: {len(batch.candidates):,} candidates', flush=True)
    output.mkdir(parents=True, exist_ok=True)
    pd.concat(observations,ignore_index=True).to_parquet(output/'observations.parquet',index=False)
    pd.concat(candidates,ignore_index=True).to_parquet(output/'candidates.parquet',index=False)
    shutil.copy2(Path(__file__).resolve().parents[1]/'feature_schema.json',output/'feature_schema.json')
    (output/'preparation.json').write_text(json.dumps({'partition':partition,'trip_ids':ids,
        'reference_routes_read':False,'search_radius_m':50,'max_candidates_per_obs':15},indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root',type=Path,required=True)
    parser.add_argument('--partition',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    prepare(args.raw_root,args.partition,args.output)
