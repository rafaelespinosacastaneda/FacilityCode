"""Facility-side scientific calculations and interchangeable selection policies.

simulation.py calls begin_round(), select(), and observe(); the policy never calls
the pool. select() uses only submitted predictions, public features, and previously
revealed observations. observe() receives newly revealed outcomes from the engine.

QualityVendiPolicy combines inverse-variance Gaussian aggregation, expected
improvement, mean stakeholder scores, and greedy RBF/PCA Vendi diversity.
Calibration filtering updates which agents contribute predictions, not their
uncertainty values. RandomFacilityPolicy implements the same interface.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from config import Selection
from submission_processing import minmax_scale

def combine_gaussians_independent(mu, sigma, sigma_floor=1e-5):
    """Combine independent Gaussian estimates using precision weighting.
    
    mu and sigma are aligned 1D arrays for one candidate across agents. Returns
    combined (mean,std). Shared-data correlations are not modeled, so combined
    uncertainty can be overconfident. A positive floor prevents infinite weights.
    """
    mu, sigma = np.asarray(mu, float), np.asarray(sigma, float)
    if len(mu) == 0 or mu.shape != sigma.shape or not np.isfinite(mu).all() or not np.isfinite(sigma).all() or (sigma < 0).any():
        raise ValueError('Invalid Gaussian predictions.')
    # Relative precision weights avoid overflow when all scales are small.
    # Apply a configured std floor before computing precision weights.
    sigma = np.maximum(sigma, sigma_floor)
    # Equivalent to inverse-variance weighting, scaled to avoid huge absolute precisions.
    relative = (sigma.min()/sigma)**2
    return np.dot(relative, mu)/relative.sum(), sigma.min()/np.sqrt(relative.sum())

def expected_improvement(mu, sigma, y_best, xi=0.01, direction='maximize'):
    """Compute facility EI from aggregated means/std and the best observed value.
    
    Supports maximize/minimize; xi is an improvement margin in outcome units.
    This is Gaussian EI on moment-matched predictions, not exact log-normal EI.
    """
    sign = 1 if direction == 'maximize' else -1
    improvement = sign*(np.asarray(mu)-y_best)-xi
    sigma = np.asarray(sigma)
    # Standardized improvement; the denominator guard handles deterministic predictions.
    z = improvement/np.maximum(sigma, 1e-12)
    return np.where(sigma > 1e-12, improvement*norm.cdf(z)+sigma*norm.pdf(z), np.maximum(improvement, 0))

def vendi_score_from_gram(K):
    """Compute effective diversity from a positive-semidefinite similarity matrix.
    
    Normalize eigenvalues to probabilities, calculate Shannon entropy, and
    exponentiate it. Identical samples give 1; an identity matrix of size n gives n.
    """
    # Only eigenvalues are needed for entropy; the kernel is symmetric.
    w = np.linalg.eigvalsh((K+K.T)/2)
    if w.min(initial=0) < -1e-7:
        raise ValueError('Kernel is not positive semidefinite.')
    w = np.maximum(w, 0)
    if w.sum() <= 1e-12:
        return 1.0
    # Ignore zero eigenvalues so log(0) is never evaluated.
    p = w[w > 0]/w.sum()
    return float(np.exp(-np.sum(p*np.log(p))))

def q_vendi_score(K, quality_scores):
    """Multiply mean raw candidate quality by the Vendi diversity of a Gram matrix.
    
    This utility expresses the mathematical objective; the policy computes its
    parts inline so it can save the separate diversity value in diagnostics.
    """
    return float(np.mean(quality_scores))*vendi_score_from_gram(K)

def mean_absolute_calibration_error(mu, sigma, y, levels=5):
    """Measure error between nominal and empirical Gaussian interval coverage.
    
    Inputs are aligned past predictions/std and REVEALED outcomes. The grid
    includes endpoints 0 and 1. This measures calibration; it does not adjust std.
    """
    expected = np.linspace(0, 1, levels)
    # Convert nominal central coverages to Gaussian half-widths in std units.
    widths = norm.ppf((1+expected)/2)
    error = np.abs(np.asarray(y)-np.asarray(mu))
    observed = np.mean(error[:, None] <= np.asarray(sigma)[:, None]*widths[None, :], axis=0)
    return float(np.mean(np.abs(observed-expected)))

class QualityVendiPolicy:
    """Facility selector maintaining a round-specific diversity and calibration state.
    """
    def __init__(self, config):
        """Store settings; calibration history is retained across rounds for logging.
        """
        self.config = config
        self.calibration_log = []

    def begin_round(self, submissions, context):
        """Reset round state and fit a feature-only PCA representation.
        
        Called once after initial submissions. Keep issued predictions for honest
        post-measurement calibration, and make all valid agents initially eligible.
        """
        # Save predictions issued before measurement; calibration must not use hindsight fits.
        self.issued = submissions.copy()
        # Reset calibration evidence for a new round, matching the notebook within-batch logic.
        self.calibration_observations = []
        self.eligible_agents = set(submissions.agent_id)
        self.selected_qualities = []
        self.round_selected = []
        self.round_id = context.round_id
        # Fit only on known features, with IDs explicitly aligned.
        # Known candidate/observed FEATURES can be combined without exposing hidden labels.
        all_features = pd.concat([context.candidates[['candidate_id', *context.features]], context.observations[['candidate_id', *context.features]]]).drop_duplicates('candidate_id').sort_values('candidate_id')
        X = all_features[list(context.features)].to_numpy(float)
        if self.config.scale_features:
            X = StandardScaler().fit_transform(X)
        # Prevent requesting more PCA dimensions than available data can support.
        n = min(self.config.pca_components, *X.shape)
        if n < 1:
            raise ValueError('Diversity representation needs candidates and features.')
        Z = PCA(n_components=n, svd_solver='full').fit_transform(X)
        # Index PCA vectors by candidate ID so selection joins cannot shift rows.
        self.embedding = pd.DataFrame(Z, index=all_features.candidate_id)
        self.gamma = self.config.gamma if self.config.gamma is not None else 1/n

    def refresh(self, submissions, context):
        # Save refreshed, not-yet-measured predictions for future calibration.
        """Replace future issued predictions when agents resubmit within a round.
        
        The simulation controls whether refresh happens. Previously recorded
        predictions remain in calibration_observations and are not recomputed.
        """
        # Save predictions issued before measurement; calibration must not use hindsight fits.
        self.issued = submissions.copy()

    def select(self, submissions, context, batch_size):
        """Greedily select up to batch_size unique submitted available candidates.
        
        Aggregate eligible predictions -> compute EI -> normalize EI -> combine with
        mean stakeholder score -> maximize mean selected raw quality times Vendi.
        Returns Selection for engine validation, measurement and storage.
        """
        remaining = submissions[submissions.candidate_id.isin(context.candidates.candidate_id)]
        if remaining.empty:
            return Selection([], pd.DataFrame())
        rows = []
        # Aggregate rows sharing a physical candidate; different agents remain attributable.
        for candidate_id, group in remaining.groupby('candidate_id', sort=True):
            # Calibration eligibility affects prediction contributors.
            use = group[group.agent_id.isin(self.eligible_agents)]
            if use.empty:
                continue
            mu, std = combine_gaussians_independent(use.prediction, use.uncertainty, self.config.sigma_floor)
            # Match notebook: all valid stakeholder scores contribute; calibration filters prediction aggregation.
            # Stakeholder scores still average ALL valid submissions, as in the notebook design.
            rows.append((candidate_id, mu, std, group.score.mean()))
        if not rows:
            raise ValueError('No eligible model covers remaining candidates.')
        table = pd.DataFrame(rows, columns=['candidate_id', 'prediction', 'uncertainty', 'score_avg']).set_index('candidate_id', drop=False)
        y = context.observations[context.target]
        best = y.max() if context.direction == 'maximize' else y.min()
        table['ei'] = expected_improvement(table.prediction, table.uncertainty, best, self.config.xi, context.direction)
        # Keep the weight interpretation consistent each time EI is recomputed.
        table['ei_normalized'] = minmax_scale(table.ei)
        table['quality'] = self.config.ei_weight*table.ei_normalized + self.config.stakeholder_weight*table.score_avg
        # Measured samples contribute diversity geometry, not selected-batch quality.
        reference = list(dict.fromkeys(context.observations.candidate_id.tolist()))
        # Retain raw qualities selected earlier in this round across measurement stages.
        chosen, qualities, diagnostics = [], list(self.selected_qualities), []
        for _ in range(min(batch_size, len(table))):
            winner, best_value, winner_vendi = None, -np.inf, None
            for candidate_id in table.index:
                if candidate_id in chosen:
                    continue
                # Use each physical sample once in the diversity kernel, even if multiple agents proposed it.
                ids = list(dict.fromkeys(reference+chosen+[candidate_id]))
                Z = self.embedding.loc[ids].to_numpy()
                K = rbf_kernel(Z, gamma=self.gamma)
                diversity = vendi_score_from_gram(K)
                # Quality average covers current-round selections; Vendi includes measured reference geometry.
                value = np.mean(qualities+[table.at[candidate_id, 'quality']])*diversity
                # Sorted candidate IDs give deterministic zero-score and tie handling.
                if value > best_value:
                    winner, best_value, winner_vendi = candidate_id, value, diversity
            chosen.append(winner)
            # Append RAW quality, not the already multiplied objective, avoiding recursive score distortion.
            qualities.append(float(table.at[winner, 'quality']))
            row = table.loc[winner].to_dict()
            # Record both the final objective and its diversity factor for posthoc inspection.
            row.update(selection_objective=float(best_value), vendi=float(winner_vendi))
            diagnostics.append(row)
        self.selected_qualities = qualities
        self.round_selected.extend(chosen)
        return Selection(chosen, pd.DataFrame(diagnostics))

    def observe(self, observations, target):
        # Preserve the predictions issued BEFORE each outcome was revealed.
        """Evaluate issued predictions against new outcomes and filter at checkpoints.
        
        Only simulation.py supplies observations. At a checkpoint, compare relative
        MACE across agents; do not duplicate model columns or recalibrate std values.
        """
        measured = observations[['candidate_id', target]].rename(columns={target: 'observed_y'})
        # Join revealed labels to predictions that existed before those labels were available.
        evidence = self.issued.merge(measured, on='candidate_id', validate='many_to_one')
        self.calibration_observations.append(evidence)
        count = len(self.round_selected)
        # In batch mode a mid-batch checkpoint cannot fire before outcomes are revealed.
        if count not in self.config.calibration_checkpoints:
            return
        evidence = pd.concat(self.calibration_observations, ignore_index=True)
        errors = {}
        for agent_id, group in evidence.groupby('agent_id'):
            errors[agent_id] = mean_absolute_calibration_error(group.prediction, np.maximum(group.uncertainty, self.config.sigma_floor), group.observed_y, self.config.calibration_levels)
        if not errors:
            return
        names = list(errors)
        scaled = minmax_scale([errors[n] for n in names])
        # Agents lacking evidence stay eligible; relative threshold reproduces the notebook's rule.
        # Do not reject sparse-submission agents solely because no selected point tested them.
        self.eligible_agents = set(self.issued.agent_id)-set(names)
        # The notebook filter compares RELATIVE calibration error across participating models.
        self.eligible_agents.update(n for n, e in zip(names, scaled) if e <= self.config.calibration_threshold)
        for n, e in zip(names, scaled):
            self.calibration_log.append({'round_id': self.round_id, 'measurements': count, 'agent_id': n, 'mace': errors[n], 'relative_mace': float(e), 'eligible': n in self.eligible_agents})

class RandomFacilityPolicy:
    """Baseline selecting uniformly from the union of valid submitted candidates.
    """
    def __init__(self, config):
        """Own an independent random generator and an empty calibration log.
        """
        self.rng = np.random.default_rng(config.seed)
        self.calibration_log = []
    def begin_round(self, submissions, context):
        """Accept the common interface; random selection has no round setup.
        """
        pass
    def refresh(self, submissions, context):
        """Accept refreshed submissions; no fitted state needs updating.
        """
        pass
    def select(self, submissions, context, batch_size):
        """Sample unique available proposal IDs and return minimal diagnostics.
        """
        # Random baseline samples only available proposed candidates, without replacement.
        ids = sorted(set(submissions.candidate_id)&set(context.candidates.candidate_id))
        selected = self.rng.choice(ids, size=min(batch_size, len(ids)), replace=False).tolist()
        return Selection(selected, pd.DataFrame({'candidate_id': selected}))
    def observe(self, observations, target):
        """Accept shared outcomes; this baseline does not learn from them.
        """
        pass

# Register another select/begin_round/refresh/observe implementation here.
FACILITY_POLICIES = {'quality_vendi': QualityVendiPolicy, 'random': RandomFacilityPolicy}
