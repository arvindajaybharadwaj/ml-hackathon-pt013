"""Person 1 handoff reader and strict contract. Does not read raw GPS/.route files."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd

KEY = ['trip_id', 'obs_id']
CANDIDATE_KEY = KEY + ['candidate_id']
FORBIDDEN = {'trip_id', 'obs_id', 'candidate_id', 'arc_id', 'partition', 'y',
             'reference_arc_id', 'alignment_valid', 'alignment_reason', 'p_match',
             'emission_prob', 'predicted_arc_id', 'schema_version'}

@dataclass
class Dataset:
    observations: pd.DataFrame
    candidates: pd.DataFrame
    labels: pd.DataFrame
    schema: dict

    def subset(self, trips):
        keep = set(trips)
        return Dataset(*(df[df.trip_id.isin(keep)].copy() for df in
                         (self.observations, self.candidates, self.labels)),
                       self.schema)

    def partition(self, name):
        return self.subset(self.observations.loc[self.observations.partition == name, 'trip_id'])

    def targets(self):
        joined = self.candidates.merge(self.labels, on=KEY, validate='many_to_one')
        joined = joined[joined.alignment_valid & joined.reference_arc_id.notna()].copy()
        joined['y'] = (joined.arc_id == joined.reference_arc_id).astype(int)
        return joined

    def validate(self, require_split=True):
        features = self.schema.get('feature_names', [])
        if (not features or len(set(features)) != len(features)
                or set(features) & FORBIDDEN or not self.schema.get('schema_version')):
            raise ValueError('Schema needs version and unique, non-identifier numeric feature_names')
        for df, required, keys in [
            (self.observations, KEY + ['partition'], KEY),
            (self.candidates, CANDIDATE_KEY + ['arc_id'] + features, CANDIDATE_KEY),
            (self.labels, KEY + ['reference_arc_id', 'alignment_valid'], KEY)]:
            if not set(required) <= set(df):
                raise ValueError(f'Missing columns: {set(required)-set(df)}')
            if df[keys].isna().any().any() or df.duplicated(keys).any():
                raise ValueError(f'Null/duplicate identity in {keys}')
            if not df.trip_id.map(lambda x: isinstance(x, str)).all():
                raise ValueError('trip_id must be a string; preserve leading zeroes')
            if not np.isfinite(pd.to_numeric(df.obs_id, errors='coerce')).all() or (df.obs_id < 0).any():
                raise ValueError('obs_id must be a nonnegative integer')
            if not (df.obs_id % 1 == 0).all():
                raise ValueError('obs_id must be integral')
        if not set(self.observations.partition) <= {'train', 'student_test'}:
            raise ValueError('Only train/student_test are allowed in development datasets')
        if self.observations.groupby('trip_id').partition.nunique().max() != 1:
            raise ValueError('A trip cannot cross partitions')
        if require_split and set(self.observations.partition) != {'train', 'student_test'}:
            raise ValueError('Both train and student_test partitions are required')
        if not self.labels.alignment_valid.map(lambda x: isinstance(x, (bool, np.bool_))).all():
            raise ValueError('alignment_valid must contain booleans')
        if (self.labels.alignment_valid & self.labels.reference_arc_id.isna()).any():
            raise ValueError('Valid alignment requires a reference arc')
        for name in ['arc_id', 'candidate_id']:
            v = pd.to_numeric(self.candidates[name], errors='coerce')
            if not np.isfinite(v).all() or (v < 0).any() or (v % 1 != 0).any():
                raise ValueError(f'{name} must be nonnegative integers')
        refs = self.labels.reference_arc_id.dropna()
        if not np.isfinite(refs).all() or (refs < 0).any() or (refs % 1 != 0).any():
            raise ValueError('Reference arcs must be nonnegative integers or null')
        obs_keys = pd.MultiIndex.from_frame(self.observations[KEY])
        if not pd.MultiIndex.from_frame(self.labels[KEY]).equals(obs_keys):
            # Order may differ, but exactly one label/mask is required for every observation.
            if set(map(tuple,self.labels[KEY].to_numpy())) != set(map(tuple,self.observations[KEY].to_numpy())):
                raise ValueError('Labels must cover every observation, including invalid alignments')
        if not pd.MultiIndex.from_frame(self.candidates[KEY]).isin(obs_keys).all():
            raise ValueError('Candidate references an unknown observation')
        if self.candidates.duplicated(KEY + ['arc_id']).any():
            raise ValueError('Duplicate arc within an observation candidate set')
        for f in features:
            if not pd.api.types.is_numeric_dtype(self.candidates[f]):
                raise ValueError(f'Feature {f} must be numeric')
            if np.isinf(self.candidates[f].to_numpy(dtype=float)).any():
                raise ValueError(f'Infinite feature {f}; use NaN for missing values')
        return self


def read_table(root, name):
    root = Path(root)
    if (root / f'{name}.parquet').exists():
        return pd.read_parquet(root / f'{name}.parquet')
    return pd.read_csv(root / f'{name}.csv', dtype={'trip_id':str})


def load_dataset(root):
    root = Path(root)
    return Dataset(
        *(read_table(root, name) for name in ['observations', 'candidates', 'labels']),
        json.loads((root / 'feature_schema.json').read_text()),
    ).validate()
