"""Record a real epoch-wise learning curve on one original grouped CV fold."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/pathsnap-matplotlib')
import json
from pathlib import Path
import torch
from ml.data import load_dataset
from ml.models import build_model
from ml.neural import resolve_device
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data=load_dataset('prepared_data').partition('train')
fold=json.loads(Path('evidence/models/records/folds.json').read_text())[0]
training=data.subset(fold['train_trips']).targets()
validation=data.subset(fold['validation_trips']).targets()
device=resolve_device('auto');torch.set_num_threads(4)
pipeline=build_model('torch_mlp',seed=13,device=device,epochs=15)
features=data.schema['feature_names']
preprocess=pipeline[:-1]
x=preprocess.fit_transform(training[features].to_numpy(dtype=float))
v=preprocess.transform(validation[features].to_numpy(dtype=float))
model=pipeline.named_steps['model'].fit(x,training.y,validation_data=(v,validation.y))
record={'purpose':'Diagnostic learning curve only; no new model selection or replacement of the exported model',
        'device':device,'fold':fold,'epochs':15,'training_loss':model.loss_curve_,
        'validation_loss':model.validation_loss_curve_}
Path('evidence/models/records/neural_fold_learning_curve.json').write_text(json.dumps(record,indent=2)+'\n')
fig,ax=plt.subplots(figsize=(7,3.7))
ax.plot(range(1,16),model.loss_curve_,'o-',label='Training trips',color='#244866')
ax.plot(range(1,16),model.validation_loss_curve_,'s-',label='Held out training trips',color='#b0723b')
ax.set(xlabel='Epoch',ylabel='Binary cross entropy',title='Neural network learning curve on grouped fold one')
ax.legend();fig.tight_layout();fig.savefig('evidence/models/plots/neural_validation_loss.png',dpi=180);plt.close(fig)
p=Path('evidence/models/comparison.json');d=json.loads(p.read_text())
d['plots']=[r for r in d['plots'] if r['name']!='neural_validation_loss.png']+[{'name':'neural_validation_loss.png','title':'Neural network grouped training and validation loss'}]
p.write_text(json.dumps(d,indent=2)+'\n')
print('Saved actual training and validation histories using',device)
