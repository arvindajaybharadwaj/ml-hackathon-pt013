"""Evaluate chosen/default decoders and export actual route and map evidence."""
import json
from pathlib import Path
from dataclasses import replace
import pandas as pd
import numpy as np
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/pathsnap-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from backend.app.config import Settings
from backend.app.main import build_services
from backend.app.maps import render_route_map


def main():
    out=Path('evidence/hmm');out.mkdir(exist_ok=True)
    settings=Settings.from_env()
    if settings.partition not in {'train','student_test'}:
        raise ValueError('Report evidence must use permitted development partitions')
    trips,_,_,prediction,evaluation=build_services(settings)
    rows=[]
    for tid in trips.list_trip_ids():
        result=evaluation.evaluate(tid);rows.append(result)
        pred=prediction.predict(tid)
        pd.DataFrame({'trip_id':tid,'obs_id':range(pred['gps_observations']),
            'classifier_arc_id':pred['classifier_only_arc_ids'],'hmm_arc_id':pred['predicted_arc_ids']}).to_parquet(out/f'{tid}_predictions.parquet',index=False)
        print(f'{tid}: classifier Top-1={result["classifier_top1_accuracy"]:.4f}, HMM={result["hmm_top1_accuracy"]:.4f}',flush=True)
        if tid=='00000052':
            maps=Path('evidence/maps');maps.mkdir(exist_ok=True)
            reference=evaluation.references.load_reference_route(tid)
            (maps/'student_test_00000052.html').write_text(render_route_map(trips.load_trip(tid),pred,reference))
            (out/'example_prediction.json').write_text(json.dumps(pred,indent=2)+'\n')
    evaluated=sum(r['evaluated_observations'] for r in rows)
    summary={'partition':settings.partition,'trip_count':len(rows),'evaluated_observations':evaluated,
        'alignment_coverage':evaluated/sum(prediction.trips.metadata(r['trip_id'])['gps_observations'] for r in rows),
        'classifier_top1_accuracy':sum(r['classifier_correct_observations'] for r in rows)/evaluated,
        'hmm_top1_accuracy':sum(r['hmm_correct_observations'] for r in rows)/evaluated,
        'classifier_mean_arc_macro_f1':float(np.mean([r['classifier_observation_arc_macro_f1'] for r in rows])),
        'hmm_mean_arc_macro_f1':float(np.mean([r['hmm_observation_arc_macro_f1'] for r in rows])),
        'classifier_mean_sequence_similarity':float(np.mean([r['classifier_only_sequence_edit_similarity'] for r in rows])),
        'hmm_mean_sequence_similarity':float(np.mean([r['sequence_edit_similarity'] for r in rows])),
        'classifier_mean_continuity':float(np.mean([r['classifier_only_route_continuity'] for r in rows])),
        'hmm_mean_continuity':float(np.mean([r['hmm_route_continuity'] for r in rows])),
        'config':json.loads(Path('artifacts/model/hmm_config.json').read_text()),
        'note':'Development results after HMM tuning; not independent evaluator performance. Sequence metrics use expanded routes; continuity uses original observation pairs.'}
    (out/'evaluation.json').write_text(json.dumps({'summary':summary,'per_trip':rows},indent=2)+'\n')
    pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,dict)} for r in rows]).to_csv(out/'evaluation.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(10,3.5))
    for ax,key,title in zip(axes,['top1_accuracy','mean_sequence_similarity','mean_continuity'],['Aligned Top 1 accuracy','Expanded sequence similarity','Direct continuity']):
        vals=[summary['classifier_'+key],summary['hmm_'+key]]
        ax.bar(['Classifier','With HMM'],vals,color=['#a2b6c6','#244866']);ax.set_ylim(0,1)
        for i,value in enumerate(vals):ax.text(i,value+.02,f'{value:.3f}',ha='center',fontsize=9)
        ax.set_title(title,fontsize=10)
    fig.tight_layout();fig.savefig(out/'decoder_comparison.png',dpi=180);plt.close(fig)
    tuning=pd.read_csv(out/'tuning.csv');fig,ax=plt.subplots(figsize=(8,3.5))
    ax.plot(range(1,len(tuning)+1),tuning.aligned_top1_accuracy,'o-',color='#244866')
    ax.set(xlabel='Configuration rank',ylabel='Aligned observation Top 1 accuracy',title='Decoder settings ranked on student test development data')
    fig.tight_layout();fig.savefig(out/'decoder_tuning.png',dpi=180);plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
