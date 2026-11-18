"""Export reproducible comparison figures and tracked evidence from a completed run."""
import argparse
import json
import shutil
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/pathsnap-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from ml.data import load_dataset
from ml.predictor import load_model
from ml.train import measure

LABELS = {'logistic_regression': 'Logistic regression', 'gaussian_nb': 'Gaussian Naive Bayes',
          'decision_tree': 'Decision tree', 'random_forest': 'Random forest',
          'hist_gradient_boosting': 'Histogram gradient boosting', 'torch_mlp': 'Neural network',
          'hist_gradient_boosting__tuned': 'Histogram boosting tuned'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=Path('runs/modal-final-20261118'))
    p.add_argument('--output', type=Path, default=Path('evidence/models'))
    args = p.parse_args()
    root, out = args.run, args.output
    plots, records = out / 'plots', out / 'records'
    plots.mkdir(parents=True, exist_ok=True); records.mkdir(exist_ok=True)
    for name in ['comparison.csv', 'configurations.json', 'folds.json', 'cv_metrics.json',
                 'metrics.json', 'manifest.json', 'tuning_trials.csv', 'tuning.sqlite3']:
        shutil.copy2(root/name, records/name)
    frame = pd.read_csv(root/'comparison.csv')
    manifest = json.loads((root/'manifest.json').read_text())
    data = load_dataset('prepared_data').partition('student_test')
    names = [LABELS[n] for n in frame.model]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.facecolor': 'white'})
    specifications = [
        ('cv_candidate_macro_f1', 'Cross validation candidate Macro F1', 'cv_macro_f1.png'),
        ('student_test_top1_accuracy', 'Student test observation Top 1 accuracy', 'test_top1.png'),
        ('student_test_candidate_macro_f1', 'Student test candidate Macro F1', 'test_macro_f1.png'),
        ('student_test_candidate_log_loss', 'Student test candidate log loss', 'log_loss.png'),
        ('student_test_inference_seconds', 'Candidate scoring time on student test in seconds', 'inference_time.png'),
        ('fit_seconds', 'Full training fit time in seconds', 'training_time.png')]
    for field, title, filename in specifications:
        fig, ax = plt.subplots(figsize=(8.6, 4.5))
        values = frame[field].values
        bars = ax.barh(names, values, color=['#244866' if '__tuned' in n else '#7c99af' for n in frame.model])
        if field == 'cv_candidate_macro_f1':
            ax.errorbar(values, names, xerr=frame.cv_candidate_macro_f1_std, fmt='none', ecolor='#222', capsize=3)
        for bar, val in zip(bars, values):
            ax.text(val + max(values)*.015, bar.get_y()+bar.get_height()/2, f'{val:.4f}', va='center', fontsize=9)
        ax.set_xlim(0, 1.08 if 'f1' in field or 'accuracy' in field else max(values)*1.23)
        ax.invert_yaxis(); ax.set_title(title, fontsize=12)
        ax.grid(axis='x', alpha=.18); ax.set_axisbelow(True)
        fig.tight_layout(); fig.savefig(plots/filename, dpi=180); plt.close(fig)
    x = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for offset, field, label, color in [(-.25, 'train_candidate_macro_f1', 'Training', '#b7c7d4'),
        (0, 'cv_candidate_macro_f1', 'Cross validation', '#6e90aa'),
        (.25, 'student_test_candidate_macro_f1', 'Student test', '#244866')]:
        ax.bar(x+offset, frame[field], width=.25, label=label, color=color)
    ax.set_xticks(x, ['LR', 'NB', 'Tree', 'Forest', 'Boost', 'MLP', 'Boost tuned'])
    ax.set_ylim(0,1); ax.set_ylabel('Binary candidate Macro F1'); ax.legend(loc='lower right')
    ax.set_title('Training and validation comparison'); fig.tight_layout()
    fig.savefig(plots/'generalisation.png', dpi=180); plt.close(fig)
    loss = json.loads((root/'torch_mlp/training_loss.json').read_text())
    fig, ax = plt.subplots(figsize=(7,3.8)); ax.plot(range(1,len(loss)+1),loss,marker='o',color='#244866')
    ax.set(xlabel='Epoch',ylabel='Binary cross entropy', title='Neural network full training loss')
    ax.text(.02,.03,'Training history only',transform=ax.transAxes, fontsize=9)
    fig.tight_layout(); fig.savefig(plots/'neural_training_loss.png',dpi=180); plt.close(fig)
    (records/'neural_training_loss.json').write_text(json.dumps(loss,indent=2)+'\n')
    trials = pd.read_csv(root/'tuning_trials.csv').sort_values('number')
    fig, ax = plt.subplots(figsize=(7,3.6))
    ax.plot(trials.number+1, trials.value, 'o', label='Trial CV F1', color='#7c99af')
    ax.plot(trials.number+1, trials.value.cummax(), '-', label='Best so far', color='#244866')
    ax.set(xlabel='Trial', ylabel='Grouped CV candidate Macro F1', title='Bayesian search history')
    ax.legend(); fig.tight_layout(); fig.savefig(plots/'bayesian_search.png',dpi=180); plt.close(fig)
    metrics = []
    for row in frame.astype(object).where(pd.notna(frame), None).to_dict('records'):
        name = row['model']
        directory = root/'model' if '__tuned' in name else root/name/'model'
        predictor = load_model(directory)
        values, scores, predictions = measure(predictor, data)
        target = data.targets().merge(scores[['trip_id','obs_id','candidate_id','p_match']],
                                     on=['trip_id','obs_id','candidate_id'], validate='one_to_one')
        matrix = confusion_matrix(target.y, target.p_match.ge(.5), labels=[0,1])
        fig, ax = plt.subplots(figsize=(4.2,3.5)); ax.imshow(matrix,cmap='Blues')
        for i in range(2):
            for j in range(2): ax.text(j,i,f'{matrix[i,j]:,}',ha='center',va='center',color='white' if matrix[i,j]>matrix.max()/2 else 'black')
        ax.set_xticks([0,1],['Non match','Match']); ax.set_yticks([0,1],['Non match','Match'])
        ax.set(xlabel='Predicted class',ylabel='Reference class',title=LABELS[name])
        fig.tight_layout(); fig.savefig(plots/f'{name}_confusion.png',dpi=160); plt.close(fig)
        # Direct adjacency measured per original observation pair, including missing predictions.
        continuous = pairs = 0
        for tid, trip in predictions.groupby('trip_id'):
            arcs = np.loadtxt(Path('Student folder/student_test')/tid/f'{tid}.arcs',dtype=int,ndmin=2)
            ids = trip.sort_values('obs_id').predicted_arc_id.to_numpy()
            for a,b in zip(ids,ids[1:]):
                pairs += 1
                if pd.notna(a) and pd.notna(b) and (a==b or arcs[int(a),1]==arcs[int(b),0]): continuous+=1
        metrics.append({**row,'label':LABELS[name], 'candidate_confusion_matrix':matrix.tolist(),
                        'classifier_direct_continuity':continuous/pairs if pairs else None})
    summary = {'run_id':root.name,'selected_model':manifest['selected_model'],
               'selected_tuned':manifest['selected_tuned'],'data_partition':'student_test',
               'training_trips':len(manifest['train_trips']),'student_test_trips':len(manifest['student_test_trips']),
               'selection_metric':manifest['selection_metric'], 'models':metrics,
               'plots':[{'name':name,'title':title} for _,title,name in specifications]+[
                   {'name':'generalisation.png','title':'Training and validation comparison'},
                   {'name':'neural_training_loss.png','title':'Neural network training history'},
                   {'name':'bayesian_search.png','title':'Bayesian optimisation history'}]}
    if (plots/'neural_validation_loss.png').is_file():
        summary['plots'].append({'name':'neural_validation_loss.png','title':'Neural network grouped training and validation loss'})
    (out/'comparison.json').write_text(json.dumps(summary,indent=2,allow_nan=False,default=lambda v:None)+'\n')
    # CSV reader uses NaN for absent tuned timing; JSON must contain null rather than NaN.
    print(f'Generated {len(metrics)} model comparisons in {out}')


if __name__=='__main__': main()
