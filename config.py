"""Shared contracts used throughout the experiment.

main.py constructs these settings; simulation.py builds Context snapshots;
agents.py returns Prediction internally; facility_policies.py returns Selection.
Dataclasses are containers, not algorithms. Keeping the shared containers here
prevents the agents and facility from needing to import each other's internals.
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

@dataclass
class Prediction:
    """Model output for aligned candidate rows.
    
    mean and std are 1D arrays of shape (n_candidates,). Inside a surrogate they
    refer to transformed targets; after ModelAgent conversion they use original
    outcome units. score_upper optionally carries exp(log_mean + log_std).
    """
    # One value per candidate; units depend on whether this is latent or converted output.
    mean: np.ndarray
    # Standard deviation, never an interval width or variance.
    std: np.ndarray
    # Optional optimistic bound used by the original maximization proposal rule.
    score_upper: np.ndarray | None = None

@dataclass
class Context:
    """Public snapshot passed by simulation.py to an agent or facility.
    
    candidates contains candidate_id and features but no unmeasured target.
    observations contains measured IDs, features and outcomes. selected_in_round
    tracks current-round measurements, while observations includes all history.
    """
    # Zero-based submission round, shared by all agents in that cycle.
    round_id: int
    # Available IDs/features; the hidden target must not appear here.
    candidates: pd.DataFrame
    # Public training history, including measured feature values and target.
    observations: pd.DataFrame
    # Explicit feature ordering prevents accidental use of target/index columns.
    features: tuple[str, ...]
    # Name of the single outcome column modeled in this first implementation.
    target: str = 'y'
    # Optimization direction; higher acquisition scores still always mean preferred.
    direction: str = 'maximize'
    # IDs measured in this submission round; history across rounds lives in observations.
    selected_in_round: tuple[str, ...] = ()

@dataclass
class Selection:
    """Facility response consumed by simulation.py.
    
    candidate_ids identifies physical measurements; diagnostics stores the
    associated score components for storage.py and posthoc analysis.
    """
    # Physical experiment IDs to reveal through pool.evaluate().
    candidate_ids: list[str]
    # Per-selection scores saved by storage.py for later explanations/analysis.
    diagnostics: pd.DataFrame

@dataclass
class AgentConfig:
    """Settings for one independent stakeholder instance.
    
    model and rule select registry entries in agents.py. parameters configures the
    surrogate constructor. proposal_limit=None submits all available candidates.
    """
    # Unique stakeholder identity, preserved through submission and selection records.
    agent_id: str
    # Lookup key in agents.MODELS for model-based agents; legacy model=csv still works.
    model: str = 'gp'
    # Lookup key in agents.RULES; independent of the predictive model choice.
    rule: str = 'upper'
    # Component-specific reproducibility setting; not automatically propagated from experiment seed.
    seed: int = 42
    # Exploration multiplier used by an upper-confidence proposal rule.
    beta: float = 1.0
    # Required improvement margin, in original outcome units.
    xi: float = 0.01
    # None scores/submits the entire remaining pool; an integer keeps a shortlist.
    proposal_limit: int | None = None
    # Keyword arguments for the selected surrogate constructor; fresh dict per agent.
    parameters: dict = field(default_factory=dict)

    # Whole-agent registry key. None preserves old model/csv configuration behavior.
    agent_type: str | None = None

@dataclass
class MetricConfig:
    """Select a registered analysis function and pass its optional keyword arguments."""
    name: str
    parameters: dict = field(default_factory=dict)

@dataclass
class MetricResult:
    """Outputs of one metric: named CSV tables and JSON-compatible summary fields.

    Metric functions return data, not file writes. analyze_run() owns persistence.
    Table keys are filename stems, e.g. best_observed_curve, without extensions.
    """
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    summary: dict = field(default_factory=dict)

@dataclass
class AnalysisContext:
    """Public recorded run data supplied equally to every selected metric.

    tables maps record names to DataFrames. Metrics can also read additional
    artifacts from directory when their calculation requires them.
    """
    directory: object
    configuration: dict
    tables: dict[str, pd.DataFrame]


@dataclass
class FacilityConfig:
    """Scientific choices for facility_policies.py.
    
    Weights combine normalized EI and stakeholder score. PCA/RBF settings define
    diversity. Calibration checkpoints count NEW measurements within each round.
    """
    # Lookup key in facility_policies.FACILITY_POLICIES.
    policy: str = 'quality_vendi'
    # Facility weight applied AFTER EI has been normalized across eligible candidates.
    ei_weight: float = 0.5
    # Facility weight for the average per-agent normalized submission score.
    stakeholder_weight: float = 0.5
    # Required improvement margin, in original outcome units.
    xi: float = 0.01
    # Requested representation dimension; bounded by available rows/features at runtime.
    pca_components: int = 3
    # RBF similarity bandwidth; None means 1 divided by actual PCA dimension.
    gamma: float | None = None
    # False preserves unscaled notebook PCA; True standardizes feature units first.
    scale_features: bool = False
    # Minimum allowed std in precision weights and Gaussian coverage calculations.
    sigma_floor: float = 1e-5
    # Counts of new measurements within a round at which model eligibility can change.
    calibration_checkpoints: tuple[int, ...] = (10, 20)
    # Keep prediction contributors at/below this RELATIVE min-max calibration error.
    calibration_threshold: float = 0.5
    # Number of nominal interval coverage levels used in MACE.
    calibration_levels: int = 5
    # Component-specific reproducibility setting; not automatically propagated from experiment seed.
    seed: int = 42

@dataclass
class ExperimentConfig:
    """Top-level configuration assembled in main.py and consumed by the engine.
    
    Paths describe inputs and outputs; features fixes column order. Agents and
    facility settings compose the experiment without changing simulation.py.
    """
    # CSV containing parameters plus retrospectively known, initially hidden outcomes.
    pool_path: str
    # Explicit feature ordering prevents accidental use of target/index columns.
    features: tuple[str, ...]
    # Historical measured CSVs, or empty to select initial_size random pool candidates.
    initial_paths: tuple[str, ...]
    # Must be a new directory, so two experiments cannot overwrite each other.
    output_directory: str
    # One entry per independent stakeholder; repeat model families freely with unique IDs.
    agents: list[AgentConfig]
    # Nested allocation settings; default_factory creates a separate settings object.
    facility: FacilityConfig = field(default_factory=FacilityConfig)
    # Name of the single outcome column modeled in this first implementation.
    target: str = 'y'
    # Name of the input ID column; pool.py standardizes it to candidate_id.
    id_column: str = 'candidate_id'
    # Optimization direction; higher acquisition scores still always mean preferred.
    direction: str = 'maximize'
    # How many times to request a fresh set of stakeholder proposals.
    rounds: int = 1
    # Maximum new measurements per submission round, not per agent.
    batch_size: int = 26
    # sequential: reveal each result; staged: reveal at checkpoints; batch: reveal at end
    # Controls when outcomes become visible to the facility and future proposals.
    measurement_mode: str = 'sequential'
    # False freezes proposals within a round; True retrains/resubmits at each stage.
    refresh_agents_each_stage: bool = False
    # Component-specific reproducibility setting; not automatically propagated from experiment seed.
    seed: int = 42

    # Used only when initial_paths is empty: reveal this many random pool labels.
    initial_size: int = 26

    # Analysis functions to execute; an empty list explicitly disables analysis.
    analysis_metrics: list[MetricConfig] = field(default_factory=lambda: [
        MetricConfig('measurement_count'), MetricConfig('best_observed'),
        MetricConfig('proposal_statistics'),
    ])
    # Optional importable modules register extra agents/metrics through decorators.
    extension_modules: list[str] = field(default_factory=list)
    # Set by load_config() so agent factories can resolve relative input resources.
    base_directory: str = ''

def metric_configs(items):
    """Accept metric names, dictionaries or MetricConfig objects from JSON/Python."""
    if not isinstance(items, (list, tuple)):
        raise ValueError('analysis_metrics must be a list, including [] to disable it.')
    return [item if isinstance(item, MetricConfig) else
            MetricConfig(item) if isinstance(item, str) else MetricConfig(**item)
            for item in items]

def load_extensions(module_names):
    """Import trusted Python modules for their agent/metric registration effects.

    Modules must be on Python's import path, e.g. in the project directory.
    This executes Python code and is intended only for your own trusted extensions.
    Called before factory lookup and before independent posthoc analysis.
    """
    import importlib
    for name in module_names:
        importlib.import_module(name)
