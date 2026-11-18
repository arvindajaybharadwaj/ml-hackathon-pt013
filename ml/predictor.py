"""Backend adapter. No reference labels are accepted by score_candidates()."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from .data import KEY, CANDIDATE_KEY

class Predictor:
    def __init__(self, pipeline, schema, metadata):
        self.pipeline, self.schema, self.metadata = pipeline, schema, metadata

    def score_candidates(self, batch):
        if isinstance(batch, pd.DataFrame):
            raise ValueError('Pass {candidates, schema_version}, not an unversioned DataFrame')
        df = batch['candidates'] if isinstance(batch,dict) else batch.candidates
        version = batch['schema_version'] if isinstance(batch,dict) else batch.schema['schema_version']
        if version != self.schema['schema_version']:
            raise ValueError('Feature schema version mismatch')
        missing = set(CANDIDATE_KEY+['arc_id']+self.schema['feature_names'])-set(df)
        if missing: raise ValueError(f'Missing predictor fields: {missing}')
        if df[CANDIDATE_KEY].isna().any().any() or df.duplicated(CANDIDATE_KEY).any():
            raise ValueError('Invalid candidate identity')
        out = df[CANDIDATE_KEY+['arc_id']].copy()
        if len(df):
            x = df[self.schema['feature_names']].to_numpy(dtype=float)
            if np.isinf(x).any(): raise ValueError('Infinite inference features')
            positive = list(self.pipeline.classes_).index(1)
            out['p_match'] = np.clip(self.pipeline.predict_proba(x)[:,positive],1e-9,1-1e-9)
            out['emission_prob'] = out.p_match / out.groupby(KEY).p_match.transform('sum')
        else:
            out['p_match'] = pd.Series(dtype=float)
            out['emission_prob'] = pd.Series(dtype=float)
        return out

    def save(self, directory):
        directory = Path(directory); directory.mkdir(parents=True,exist_ok=True)
        joblib.dump(self.pipeline,directory/'model.joblib')
        (directory/'feature_schema.json').write_text(json.dumps(self.schema,indent=2)+'\n')
        (directory/'metadata.json').write_text(json.dumps(self.metadata,indent=2)+'\n')


def load_model(artifact_dir):
    """Only load artifacts from a trusted teammate: joblib executes pickle code."""
    root = Path(artifact_dir)
    metadata = json.loads((root/'metadata.json').read_text())
    return Predictor(joblib.load(root/'model.joblib'),
                     json.loads((root/'feature_schema.json').read_text()),metadata)
