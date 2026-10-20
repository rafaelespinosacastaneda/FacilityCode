"""Boundary between stakeholder submissions and the facility.

simulation.py calls process_submissions() after collecting agent DataFrames.
This module normalizes names and scores, validates candidate identity, and returns
accepted rows plus a rejection report. It never queries hidden outcomes or makes
facility selections. facility_policies.py also reuses minmax_scale().
"""
import numpy as np
import pandas as pd

def minmax_scale(values):
    """Map a finite numeric vector to [0,1]; constant vectors become zeros.
    
    Used for stakeholder scores, facility EI and relative calibration errors.
    The constant-case convention avoids division by zero.
    """
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Cannot normalize nonfinite values.')
    span = np.ptp(values) if len(values) else 0.0
    return (values - values.min()) / span if span > 0 else np.zeros_like(values)

def process_submissions(frames, valid_ids):
    """Validate agent tables and return (accepted_rows, rejected_rows).
    
    Input: list of DataFrames and available candidate IDs. Output columns include
    agent_id, candidate_id, prediction, uncertainty, normalized score, raw_score.
    Finite nonnegative uncertainty is required; zero is floored by the facility.
    Scores are normalized separately for each agent within each input frame.
    """
    accepted, rejected = [], []
    valid = set(map(str, valid_ids))
    for original in frames:
        f = original.copy()
        f.columns = f.columns.str.strip().str.lower()
        # Accept the original notebook export spellings, then use one internal schema.
        f = f.rename(columns={'pool_id': 'candidate_id', 'index': 'candidate_id', 'std': 'uncertainty'})
        if f.columns.duplicated().any():
            raise ValueError('Ambiguous submission columns after alias mapping.')
        # The common contract allows arbitrary model families to submit the same table.
        required = ['agent_id', 'candidate_id', 'prediction', 'uncertainty', 'score']
        if not set(required).issubset(f.columns):
            raise ValueError(f'Submission needs {required}')
        if f[['agent_id', 'candidate_id']].isna().any().any():
            raise ValueError('Missing agent or candidate ID.')
        f['candidate_id'] = f.candidate_id.astype(str)
        f['agent_id'] = f.agent_id.astype(str)
        if f.duplicated(['agent_id', 'candidate_id']).any():
            raise ValueError('Duplicate candidate within agent submission.')
        for col in ['prediction', 'uncertainty', 'score']:
            f[col] = pd.to_numeric(f[col], errors='coerce')
        # Track row-level rejection reasons so invalid submissions remain inspectable.
        reasons = np.full(len(f), '', dtype=object)
        reasons[~f.candidate_id.isin(valid)] = 'unavailable_candidate'
        # Reject nonfinite predictions/scores/std before any downstream aggregation.
        numeric_ok = np.isfinite(f[['prediction', 'uncertainty', 'score']].to_numpy()).all(axis=1)
        reasons[~numeric_ok] = 'nonfinite_value'
        reasons[(f.uncertainty < 0).to_numpy()] = 'negative_uncertainty'
        bad = f.loc[reasons != ''].copy()
        # Rejected rows are returned separately for storage; they do not reach the facility.
        bad['reason'] = reasons[reasons != '']
        rejected.append(bad)
        f = f.loc[reasons == ''].copy()
        # Retain pre-normalization values to make stakeholder scaling auditable.
        f['raw_score'] = f.score
        # Normalize within a stakeholder; different score units cannot be averaged directly.
        for _, index in f.groupby('agent_id').groups.items():
            f.loc[index, 'score'] = minmax_scale(f.loc[index, 'score'])
        accepted.append(f[required + ['raw_score']])
    if not accepted:
        raise ValueError('No submissions.')
    # Long-table layout adds agents as rows rather than adding numbered model columns.
    good = pd.concat(accepted, ignore_index=True)
    if good.duplicated(['agent_id', 'candidate_id']).any():
        raise ValueError('Agent IDs must be unique across submitted frames.')
    if good.empty:
        raise ValueError('No valid submission rows.')
    return good, pd.concat(rejected, ignore_index=True)
