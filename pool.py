"""Own the retrospective candidate CSV and its hidden outcome column.

main.py creates CSVPool. simulation.py asks for feature-only candidates and calls
evaluate() AFTER facility selection. Agents and facility receive Context objects,
not the pool object. This is an architectural information boundary, not a security
sandbox against deliberately malicious Python code.

Preparation helpers at the bottom convert the original notebook CSV collection
into canonical pool and observation tables before the simulation starts.
"""
from pathlib import Path
import numpy as np
import pandas as pd

def read_table(path, id_column='candidate_id'):
    """Load a CSV while preserving explicit ID strings, including leading zeros.
    
    Feature and target columns keep pandas numeric inference. Strip surrounding
    column-name whitespace to accommodate exported notebook files.
    """
    frame = pd.read_csv(path, dtype={id_column: str})
    frame.columns = frame.columns.str.strip()
    return frame

class CSVPool:
    """Store the full labeled table privately and expose feature-only candidates.
    """
    def __init__(self, config):
        """Load and validate the configured CSV; index it by stable candidate_id.
        
        The original ID column is renamed internally so downstream code shares one
        canonical schema. This retrospective pool requires finite known outcomes.
        """
        self.config = config
        frame = read_table(config.pool_path, config.id_column)
        if config.id_column not in frame:
            raise ValueError('Pool needs an explicit stable ID column; do not use filtered row positions.')
        frame = frame.rename(columns={config.id_column: 'candidate_id'})
        self._check(frame)
        if not np.isfinite(frame[config.target].to_numpy(float)).all():
            raise ValueError('Retrospective simulation requires finite pool outcomes.')
        # Stable-ID indexing allows .loc selection without relying on row positions.
        self._table = frame.set_index('candidate_id', drop=False)

    def _check(self, frame):
        """Validate required columns, IDs and numeric features; normalize IDs in place.
        """
        required = ['candidate_id', *self.config.features, self.config.target]
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f'Missing columns: {missing}')
        if frame.candidate_id.isna().any():
            raise ValueError('Missing candidate IDs.')
        # Normalize ID type before joins; read_table preserves leading-zero strings.
        frame['candidate_id'] = frame.candidate_id.astype(str)
        if frame.candidate_id.duplicated().any():
            raise ValueError('Duplicate candidate IDs.')
        if not np.isfinite(frame[list(self.config.features)].to_numpy(float)).all():
            raise ValueError('Features must be finite numeric values.')

    def initial_observations(self):
        """Load historical observations, or reveal a random seed set from the pool.
        
        Reject duplicate IDs and inconsistent labels/features for IDs also present
        in the pool. Initial candidates may legitimately lie outside the search pool.
        With no historical paths, initial_size and seed choose a fresh campaign.
        """
        frames = []
        for path in self.config.initial_paths:
            f = read_table(path, self.config.id_column).rename(columns={self.config.id_column: 'candidate_id'})
            self._check(f)
            if not np.isfinite(f[self.config.target].to_numpy(float)).all():
                raise ValueError('Initial labels must be finite.')
            frames.append(f[['candidate_id', *self.config.features, self.config.target]])
        if not frames:
            # A fresh campaign needs a measured seed set before any model can fit.
            # Sample IDs without accessing their y values, then reveal only those IDs.
            size = self.config.initial_size
            if not isinstance(size, int) or isinstance(size, bool) or not 4 <= size < len(self._table):
                raise ValueError('initial_size must be an integer >=4 and smaller than the pool.')
            rng = np.random.default_rng(self.config.seed)
            ids = rng.choice(self._table.index.to_numpy(), size=size, replace=False).tolist()
            return self.evaluate(ids)
        result = pd.concat(frames, ignore_index=True)
        if result.candidate_id.duplicated().any():
            raise ValueError('Initial history contains duplicate candidates.')
        # Verify overlapping initial observations describe the same physical candidates.
        overlap = result[result.candidate_id.isin(self._table.index)]
        for _, row in overlap.iterrows():
            cols = [*self.config.features, self.config.target]
            if not np.allclose(row[cols].to_numpy(float), self._table.loc[row.candidate_id, cols].to_numpy(float)):
                raise ValueError(f'Initial observation differs from pool: {row.candidate_id}')
        return result

    def candidates(self, observed_ids=()):
        """Return a copy of unmeasured IDs and features, excluding all observed IDs.
        
        No target column is returned. simulation.py puts this table into Context.
        """
        # Exclude observed IDs before constructing the public candidate view.
        f = self._table.loc[~self._table.index.isin(observed_ids)]
        return f[['candidate_id', *self.config.features]].reset_index(drop=True).copy()

    def evaluate(self, candidate_ids):
        """Reveal original-scale outcomes for selected valid IDs in selection order.
        
        Called by the engine after a decision, or to reveal random initial IDs.
        This method does not track
        measurement history itself. The engine prevents repeated measurements.
        """
        ids = list(candidate_ids)
        if len(ids) != len(set(ids)) or not set(ids).issubset(self._table.index):
            raise ValueError('Invalid or duplicate measurement IDs.')
        return self._table.loc[ids, ['candidate_id', *self.config.features, self.config.target]].reset_index(drop=True).copy()

def prepare_pool(search_path, labeled_path, output_path, features, target='y'):
    """Prepare a labeled pool by exact feature matching before running experiments.
    
    Assign stable IDs before filtering and reject ambiguous label matches.
    Returns the prepared DataFrame and writes it as a CSV.
    """
    search, labels = read_table(search_path), read_table(labeled_path)
    # Remove any search-space target so the authoritative labeled source supplies it.
    search = search.drop(columns=[target], errors='ignore')
    if 'candidate_id' not in search:
        search.insert(0, 'candidate_id', [f'pool_{i}' for i in range(len(search))])
    if labels.duplicated(list(features)).any():
        raise ValueError('Labeled source has ambiguous duplicate feature rows.')
    # Use one validated join instead of repeatedly scanning all labels row by row.
    result = search.merge(labels[[*features, target]], on=list(features), how='left', validate='many_to_one')
    if result[target].isna().any():
        raise ValueError('Some search candidates have no outcome match.')
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # Write ordinary CSV without a pandas row-index column.
    result.to_csv(output_path, index=False)
    return result

def prepare_initial_history(batch_paths, prepared_pool_path, output_path, features, target='y'):
    """Merge old batches and attach canonical IDs by exact feature values.
    
    Candidates outside the search pool get separate initial IDs. Repeated
    parameter combinations require explicit replicate handling and are rejected.
    """
    pool = read_table(prepared_pool_path)
    keys = list(features)
    if pool.duplicated(keys).any():
        raise ValueError('Cannot resolve feature-based IDs: pool contains duplicate parameter combinations.')
    # Only feature values and canonical IDs participate in resolving historical batches.
    mapping = pool[['candidate_id', *keys]]
    frames = []
    for batch_number, path in enumerate(batch_paths):
        f = read_table(path)[[*keys, target]].copy()
        f = f.merge(mapping, on=keys, how='left', validate='many_to_one', sort=False)
        # The original first batch can legitimately be outside First_SearchSpace.
        absent = f.candidate_id.isna()
        f.loc[absent, 'candidate_id'] = [f'initial_{batch_number}_{i}' for i in f.index[absent]]
        frames.append(f[['candidate_id', *keys, target]])
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(keys).any():
        raise ValueError('Historical batches repeat parameter combinations; resolve replicates explicitly.')
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # Write ordinary CSV without a pandas row-index column.
    result.to_csv(output_path, index=False)
    return result
