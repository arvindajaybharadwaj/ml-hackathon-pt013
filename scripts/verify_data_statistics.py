"""Recompute report statistics and diagnostic plots from real raw/exported data."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/pathsnap-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]
summary={}; intervals=[]; reasons=[]
for part in ['train','student_test']:
    obs=pd.read_parquet(root/'prepared_data'/part/'observations.parquet')
    labels=pd.read_parquet(root/'prepared_data'/part/'labels.parquet')
    candidates=pd.read_parquet(root/'prepared_data'/part/'candidates.parquet')
    gaps=[]; invalid=0
    for p in (root/'Student folder'/part).glob('*/*.track'):
        raw=np.loadtxt(p,ndmin=2); invalid+=int((~np.isfinite(raw)).sum()); gaps.extend(np.diff(raw[:,2]))
    gaps=np.asarray(gaps); intervals.extend(gaps); reasons.extend(labels.alignment_reason.tolist())
    joined=candidates[['trip_id','obs_id','arc_id']].merge(labels,on=['trip_id','obs_id'])
    joined=joined[joined.alignment_valid & joined.reference_arc_id.notna()]
    summary[part]={'trips':int(obs.trip_id.nunique()),'observations':len(obs),'candidates':len(candidates),
        'valid_aligned_observations':int(labels.alignment_valid.sum()),
        'alignment_coverage':float(labels.alignment_valid.mean()),'raw_nonfinite_values':invalid,
        'one_second_intervals':int(np.isclose(gaps,1).sum()),'intervals':len(gaps),
        'one_second_fraction':float(np.isclose(gaps,1).mean()),'max_gap_s':float(gaps.max()),
        'candidate_feature_missing_values':int(candidates.select_dtypes('number').isna().sum().sum())}
    summary[part].update(aligned_candidate_rows=len(joined),
        positive_candidate_rows=int(joined.arc_id.eq(joined.reference_arc_id).sum()),
        zero_candidate_observations=len(obs)-len(candidates[['trip_id','obs_id']].drop_duplicates()),
        longitude_range=[float(obs.lon.min()),float(obs.lon.max())],
        latitude_range=[float(obs.lat.min()),float(obs.lat.max())])
intervals=np.asarray(intervals)
summary['combined_one_second_fraction']=float(np.isclose(intervals,1).mean())
out=root/'evidence/data';out.mkdir(exist_ok=True)
(out/'verified_statistics.json').write_text(json.dumps(summary,indent=2)+'\n')
fig,ax=plt.subplots(figsize=(7,3.8));ax.hist(intervals[(intervals>=0)&(intervals<=5)],bins=50,color='#6e90aa')
ax.set(xlabel='Sampling interval in seconds',ylabel='Count',title='GPS sampling intervals up to five seconds')
ax.text(.97,.95,f'{100*summary["combined_one_second_fraction"]:.2f}% of all intervals are one second',ha='right',va='top',transform=ax.transAxes,fontsize=9)
fig.tight_layout();fig.savefig(out/'sampling_interval_dist.png',dpi=180);plt.close(fig)
counts=pd.Series(reasons).value_counts();fig,ax=plt.subplots(figsize=(8,4))
ax.bar(counts.index,counts.values,color='#6e90aa');ax.tick_params(axis='x',rotation=15)
ax.set(ylabel='Observations',title='Actual reference alignment outcomes')
fig.tight_layout();fig.savefig(out/'alignment_coverage.png',dpi=180);plt.close(fig)
print(json.dumps(summary,indent=2))
