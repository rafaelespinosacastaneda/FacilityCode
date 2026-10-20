"""Read saved run records and compute posthoc summaries.

main.py calls analyze_run() after simulation, or you can run this file directly:
python analysis.py results/experiment_001

This module does not influence acquisition decisions and does not retrain agents.
Its current measures are best observed outcome, submission/selection counts, and
acceptance rates. These are foundations for additional fairness metrics, not a
complete fairness evaluation. compare_runs() compares separately executed runs.
"""
import argparse
import sys

# CLI execution and plugins must share the SAME registry module instance.
# Otherwise a plugin importing analysis would register into a second module.
if __name__ == '__main__':
    sys.modules['analysis'] = sys.modules[__name__]

import json
from pathlib import Path
import numpy as np
import pandas as pd
from config import AnalysisContext, MetricResult, metric_configs, load_extensions

def load_optional(path):
    """Read a recorded CSV with string IDs, or return an empty table if absent.
    """
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype={'candidate_id': str, 'agent_id': str})
    except pd.errors.EmptyDataError:
        # Some metrics legitimately emit empty tables with no columns.
        return pd.DataFrame()

def best_observed_curve(initial, measurements, target='y', direction='maximize', mode='sequential'):
    """Compute incumbent outcome after each stage, including the initial incumbent.
    
    The x-axis counts new measurements, excluding the initial set. Batch stages
    produce one update at completion; mode is retained in the public signature
    but grouping by recorded stages determines the actual visibility schedule.
    """
    best_fn = max if direction == 'maximize' else min
    best = float(initial[target].max() if direction == 'maximize' else initial[target].min())
    # Include the initial incumbent at zero NEW measurements.
    rows = [{'measurements': 0, 'best_observed': best}]
    count = 0
    # Preserve batch/stage visibility: no invented within-batch learning curve.
    # Respect the recorded information schedule: one curve update per reveal stage.
    for _, stage in measurements.groupby(['round_id', 'stage_id'], sort=False):
        count += len(stage)
        best = best_fn(best, float(stage[target].max() if direction == 'maximize' else stage[target].min()))
        rows.append({'measurements': count, 'best_observed': best})
    return pd.DataFrame(rows)

def proposal_statistics(submissions, observations):
    """Count unique proposals and selected matches per agent over all rounds.
    
    Repeated stage refreshes do not count as new demand for the same round/ID.
    Multiple agents proposing one chosen candidate each receive full match credit;
    this is descriptive attribution, not fractional capacity allocation.
    """
    if submissions.empty or observations.empty:
        return pd.DataFrame()
    # Only original submission per round; repeated refreshes do not inflate demand.
    # Refreshes can repeat a proposal; count it once per agent/candidate/round.
    submitted = submissions.sort_values('stage_id').drop_duplicates(['round_id', 'agent_id', 'candidate_id'])
    chosen = observations[['round_id', 'candidate_id']].drop_duplicates().assign(selected=True)
    joined = submitted.merge(chosen, on=['round_id', 'candidate_id'], how='left')
    # Unmatched proposals are treated as unselected; matching agents all receive full credit.
    joined['selected'] = joined.selected.eq(True)
    return joined.groupby('agent_id').agg(submitted=('candidate_id', 'size'), selected_proposals=('selected', 'sum'), acceptance_rate=('selected', 'mean')).reset_index()

# Metric plugins return tables and summary fields. Only analyze_run writes files.
METRICS = {}
DEFAULT_METRICS = ['measurement_count', 'best_observed', 'proposal_statistics']

def register_metric(name):
    """Register a function/callable with signature (AnalysisContext, **parameters).

    The result must be MetricResult. A class instance with __call__ also works.
    No conditional branches in analyze_run are needed for additional metrics.
    """
    def decorate(function):
        if not isinstance(name, str) or not name or name in METRICS:
            raise ValueError(f'Invalid or duplicate metric name: {name!r}')
        if not callable(function):
            raise TypeError('A metric must be callable.')
        METRICS[name] = function
        return function
    return decorate

@register_metric('measurement_count')
def measurement_count_metric(context):
    """Count newly acquired measurements, excluding the initial seed history."""
    return MetricResult(summary={'measurements': len(context.tables['observations'])})

@register_metric('best_observed')
def best_observed_metric(context):
    """Wrap the existing stage-aware incumbent curve without changing its math."""
    initial, observed = context.tables['initial_observations'], context.tables['observations']
    cfg = context.configuration
    if initial.empty:
        return MetricResult()
    # Preserve an initial point even when no additional measurements were acquired.
    if observed.empty:
        observed = pd.DataFrame(columns=['round_id', 'stage_id', cfg['target']])
    curve = best_observed_curve(initial, observed, cfg['target'], cfg['direction'], cfg['measurement_mode'])
    return MetricResult(tables={'best_observed_curve': curve},
                        summary={'best_observed': float(curve.best_observed.iloc[-1])})

@register_metric('proposal_statistics')
def proposal_statistics_metric(context):
    """Wrap the existing full-credit proposal statistics for configurable execution."""
    stats = proposal_statistics(context.tables['submissions'], context.tables['observations'])
    return MetricResult(tables={'proposal_statistics': stats})

def analyze_run(directory, metrics=None):
    """Execute registered metrics chosen in the run config or an explicit override.

    metrics accepts names, MetricConfig objects, or dictionaries containing name
    and parameters. None uses saved analysis_metrics (or legacy defaults); []
    executes none. Unknown/duplicate names fail before any metric is executed.
    Tables and summary fields must have unique output names so metrics cannot
    silently overwrite one another. Raw experiment records are protected.
    """
    import re
    directory = Path(directory)
    cfg = json.loads((directory/'config.json').read_text())
    # Also load extensions when analyzing in a separate Python process.
    load_extensions(cfg.get('extension_modules', []))
    selected = metric_configs(cfg.get('analysis_metrics', DEFAULT_METRICS) if metrics is None else metrics)
    names = [spec.name for spec in selected]
    if len(names) != len(set(names)):
        raise ValueError('Choose each metric only once; use parameters for its settings.')
    unknown = set(names)-set(METRICS)
    if unknown:
        raise ValueError(f'Unknown metrics {sorted(unknown)}. Registered: {sorted(METRICS)}')
    # A metric can inspect any recorded CSV through context.tables, including new
    # evidence added later, without adding another loader branch to this function.
    record_paths = list(directory.glob('*.csv'))
    context = AnalysisContext(directory, cfg,
                              {path.stem: load_optional(path) for path in record_paths})
    for required in ['initial_observations', 'observations', 'submissions']:
        context.tables.setdefault(required, pd.DataFrame())
    tables, summary = {}, {}
    protected = {'initial_observations', 'observations', 'submissions',
                 'validation_report', 'selections', 'calibration_history'}
    for spec in selected:
        result = METRICS[spec.name](context, **spec.parameters)
        if not isinstance(result, MetricResult):
            raise TypeError(f'Metric {spec.name!r} must return MetricResult.')
        for name, frame in result.tables.items():
            if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', name):
                raise ValueError('Metric table names must be plain filename stems.')
            if name in protected or name.startswith('raw_submissions_agent_'):
                raise ValueError(f'Metric cannot replace raw run evidence: {name}')
            if name in tables:
                raise ValueError(f'Multiple metrics write table {name!r}.')
            if not isinstance(frame, pd.DataFrame):
                raise TypeError('Metric tables must be DataFrames.')
            tables[name] = frame
        overlap = set(summary)&set(result.summary)
        if overlap:
            raise ValueError(f'Multiple metrics produce summary keys: {sorted(overlap)}')
        summary.update(result.summary)
    # Validate JSON serialization before writing any output.
    serialized = json.dumps(summary, indent=2, allow_nan=False)
    for name, table in tables.items():
        table.to_csv(directory/f'{name}.csv', index=False)
    (directory/'analysis_summary.json').write_text(serialized, encoding='utf-8')
    return summary

def compare_runs(directories, output_path):
    """Export final summaries for separately executed facility/baseline runs.
    
    This does not align budgets or calculate significance automatically; callers
    must supply comparable experiments and repeat seeds for statistical claims.
    """
    rows = []
    for directory in directories:
        # Analyze each baseline/policy run independently before collecting its final summary.
        row = analyze_run(directory)
        row['run'] = str(directory)
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_path, index=False)


@register_metric("stakeholder_satisfaction")
def stakeholder_satisfaction(
    context,
    n_clusters=5,
    top_k=26,
    group_by_stage=False,
    pool_path=None,
    seed=0,
):
    """
    Calculate stakeholder satisfaction using the notebook's cluster rule.

    Procedure:
    1. Standardize the configured pool features.
    2. Reduce them to two PCA components.
    3. Cluster the pool with KMeans.
    4. For each stakeholder, count proposals in each selected cluster.
    5. Convert counts to scores and average over distinct selected clusters.

    Parameters
    ----------
    top_k:
        Use this many highest-scoring submitted candidates per stakeholder.
        None uses every submitted candidate.
    group_by_stage:
        False evaluates the complete selection of each round.
        True evaluates each measurement stage separately.
        Required when stakeholders refresh proposals within a round.
    pool_path:
        Optional replacement location if the original pool file has moved.
    """
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    from config import MetricResult

    cfg = context.configuration
    proposals = context.tables["submissions"].copy()
    selected = context.tables["observations"].copy()

    if selected.empty:
        return MetricResult(
            summary={"mean_stakeholder_satisfaction": None}
        )
    if proposals.empty:
        raise ValueError(
            "Satisfaction requires saved stakeholder submissions."
        )
    if cfg.get("refresh_agents_each_stage", False) and not group_by_stage:
        raise ValueError(
            "Set group_by_stage=True when agents refresh within a round."
        )
    if top_k is not None:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer or None.")
    if (
        isinstance(n_clusters, bool)
        or not isinstance(n_clusters, int)
        or n_clusters < 1
    ):
        raise ValueError("n_clusters must be a positive integer.")

    # Load the original pool for its FEATURES and stable IDs only.
    # The outcome y must not influence the clustering.
    path = Path(pool_path or cfg["pool_path"]).expanduser()
    if not path.is_absolute():
        path = Path(cfg.get("base_directory") or ".") / path

    id_column = cfg.get("id_column", "candidate_id")
    features = list(cfg["features"])

    pool = pd.read_csv(path, dtype={id_column: str})
    pool.columns = pool.columns.str.strip()

    if id_column not in pool or not set(features).issubset(pool.columns):
        raise ValueError(
            "Pool columns do not match the saved configuration."
        )
    if pool[id_column].isna().any() or pool[id_column].duplicated().any():
        raise ValueError("Pool IDs must be present and unique.")

    X = pool[features].to_numpy(float)
    if not np.isfinite(X).all() or min(X.shape) < 2:
        raise ValueError(
            "Clustering requires finite features and at least two rows/features."
        )
    if n_clusters > len(pool):
        raise ValueError("n_clusters exceeds pool size.")

    # Fit once: all stakeholders use the same PCA space and clusters.
    X_scaled = StandardScaler().fit_transform(X)
    Z = PCA(
        n_components=2,
        svd_solver="full",
    ).fit_transform(X_scaled)

    labels = KMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init=10,
    ).fit_predict(Z)

    cluster_of = pd.Series(
        labels,
        index=pool[id_column].astype(str),
    )

    # Match candidates through their original IDs, not row positions.
    for frame in (proposals, selected):
        frame["candidate_id"] = frame.candidate_id.astype(str)
        unknown = set(frame.candidate_id) - set(cluster_of.index)
        if unknown:
            raise ValueError(
                "Candidate IDs missing from analysis pool: "
                f"{sorted(unknown)[:5]}"
            )

    proposals["agent_id"] = proposals.agent_id.astype(str)

    # Include configured stakeholders even if none of their proposals survived
    # validation: their proposal counts in selected clusters will be zero.
    agent_ids = list(dict.fromkeys(
        [str(a["agent_id"]) for a in cfg.get("agents", [])]
        + proposals.agent_id.tolist()
    ))

    grouping = (
        ["round_id", "stage_id"]
        if group_by_stage
        else ["round_id"]
    )

    score_rows = []
    cluster_rows = []

    for key, measurements in selected.groupby(grouping, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        round_id = key[0]
        stage_id = key[1] if group_by_stage else None

        chosen_ids = measurements.candidate_id.drop_duplicates()

        # As in your notebook, average over distinct selected clusters.
        # A cluster selected several times still contributes one cluster score.
        chosen_clusters = sorted(
            cluster_of.loc[chosen_ids].unique()
        )

        for agent_id in agent_ids:
            submitted = proposals[
                (proposals.round_id == round_id)
                & (proposals.agent_id == agent_id)
            ]

            if group_by_stage:
                submitted = submitted[
                    submitted.stage_id <= stage_id
                ]

            if not submitted.empty:
                # Use the most recent proposals available before these
                # measurements revealed their outcomes.
                latest_stage = submitted.stage_id.max()
                submitted = submitted[
                    submitted.stage_id == latest_stage
                ]

                submitted = (
                    submitted
                    .sort_values(
                        "score",
                        ascending=False,
                        kind="stable",
                    )
                    .drop_duplicates("candidate_id")
                )

                if top_k is not None:
                    submitted = submitted.head(top_k)

            proposed_ids = submitted.candidate_id.tolist()
            counts = cluster_of.loc[proposed_ids].value_counts()

            cluster_scores = []

            for cluster_id in chosen_clusters:
                count = int(counts.get(cluster_id, 0))

                # Preserve the executable notebook rule:
                # 0->1, 1->2, 2->3, 3->4, >=4->5.
                score = min(count + 1, 5)
                cluster_scores.append(score)

                cluster_rows.append({
                    "round_id": round_id,
                    "stage_id": stage_id,
                    "agent_id": agent_id,
                    "cluster_id": int(cluster_id),
                    "proposals_in_cluster": count,
                    "cluster_score": score,
                })

            score_rows.append({
                "round_id": round_id,
                "stage_id": stage_id,
                "agent_id": agent_id,
                "n_proposals_used": len(proposed_ids),
                "n_selected": len(chosen_ids),
                "n_selected_clusters": len(chosen_clusters),
                "satisfaction": float(np.mean(cluster_scores)),
            })

    satisfaction = pd.DataFrame(score_rows)

    # Return data to the existing registry executor.
    # analyze_run() handles saving these tables and the summary.
    return MetricResult(
        tables={
            "stakeholder_satisfaction": satisfaction,
            "stakeholder_satisfaction_clusters": pd.DataFrame(
                cluster_rows
            ),
            "satisfaction_pool_clusters": pd.DataFrame({
                "candidate_id": pool[id_column].astype(str),
                "cluster_id": labels,
            }),
        },
        summary={
            "mean_stakeholder_satisfaction": float(
                satisfaction.satisfaction.mean()
            )
        },
    )



@register_metric("pca_selection_plots")
def pca_selection_plots(
    context,
    round_id=None,
    top_k=26,
    stage_id=None,
    pool_path=None,
    dpi=220,
):
    """
    Plot facility selections and stakeholder preferences in a shared PCA space.

    Parameters
    ----------
    round_id:
        Zero-based round to display. None selects the latest measured round.
    top_k:
        Number of highest-scoring proposals shown per stakeholder.
        None displays every submitted candidate.
    stage_id:
        Optional stage within the selected round.
        Required if agents refresh proposals between stages.
    pool_path:
        Override the pool location if it has moved.
    dpi:
        Resolution of the saved PNG figures.

    Outputs
    -------
    pca_round_XXX_overview.png
    pca_round_XXX_stakeholders.png
    pca_round_XXX_coordinates.csv

    When stage_id is supplied, filenames also include the stage.
    """
    from pathlib import Path
    from math import ceil

    import numpy as np
    import pandas as pd
    import matplotlib

    # Save figures without requiring a graphical desktop.
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    from config import MetricResult

    cfg = context.configuration
    observations = context.tables["observations"].copy()
    initial = context.tables["initial_observations"].copy()
    submissions = context.tables["submissions"].copy()

    if observations.empty:
        raise ValueError("No acquired measurements are available to plot.")

    if top_k is not None:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer or None.")

    # Select the round using the recorded, zero-based round IDs.
    available_rounds = sorted(observations["round_id"].unique())
    round_id = available_rounds[-1] if round_id is None else round_id

    if round_id not in available_rounds:
        raise ValueError(
            f"Round {round_id} is unavailable. "
            f"Available rounds: {available_rounds}"
        )

    if cfg.get("refresh_agents_each_stage", False) and stage_id is None:
        raise ValueError(
            "Agents refreshed within the round. Specify stage_id so the "
            "plot compares selections with the proposals available then."
        )

    round_observations = observations[
        observations["round_id"] == round_id
    ]

    if stage_id is not None:
        available_stages = sorted(
            round_observations["stage_id"].unique()
        )
        if stage_id not in available_stages:
            raise ValueError(
                f"Stage {stage_id} is unavailable. "
                f"Available stages: {available_stages}"
            )

        current = round_observations[
            round_observations["stage_id"] == stage_id
        ].copy()

        previous = observations[
            (observations["round_id"] < round_id)
            | (
                (observations["round_id"] == round_id)
                & (observations["stage_id"] < stage_id)
            )
        ].copy()
    else:
        current = round_observations.copy()
        previous = observations[
            observations["round_id"] < round_id
        ].copy()

    # Load pool IDs as strings so IDs retain their original identity.
    path = Path(pool_path or cfg["pool_path"]).expanduser()
    if not path.is_absolute():
        path = Path(cfg.get("base_directory") or ".") / path

    id_column = cfg.get("id_column", "candidate_id")
    features = list(cfg["features"])
    target = cfg["target"]
    maximize = cfg.get("direction", "maximize") == "maximize"

    pool = pd.read_csv(path, dtype={id_column: str})
    pool.columns = pool.columns.str.strip()

    required = {id_column, target, *features}
    if not required.issubset(pool.columns):
        raise ValueError(
            f"Pool is missing columns: {required - set(pool.columns)}"
        )
    if target in features or id_column in features:
        raise ValueError("Features must exclude the target and candidate ID.")
    if pool[id_column].isna().any() or pool[id_column].duplicated().any():
        raise ValueError("Pool IDs must be present and unique.")

    pool = pool.rename(columns={id_column: "candidate_id"})
    pool["candidate_id"] = pool["candidate_id"].astype(str)
    pool = pool.set_index("candidate_id", drop=False)

    X = pool[features].to_numpy(float)
    y = pool[target].to_numpy(float)

    if (
        min(X.shape) < 2
        or not np.isfinite(X).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError(
            "PCA requires finite features/outcomes and at least "
            "two candidates and two features."
        )

    # Fit once on FEATURES ONLY. Every panel uses identical geometry.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2, svd_solver="full")
    coordinates = pca.fit_transform(X_scaled)

    pc = pd.DataFrame(
        coordinates,
        index=pool.index,
        columns=["PC1", "PC2"],
    )

    # Measurements can include initial samples outside the search pool.
    # Transform them using the same fitted scaler and PCA.
    def project_measurements(frame):
        if frame.empty:
            return np.empty((0, 2))

        values = frame[features].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError("Measured features contain nonfinite values.")

        return pca.transform(scaler.transform(values))

    initial_xy = project_measurements(initial)
    previous_xy = project_measurements(previous)
    current_xy = project_measurements(current)

    # Best measured value uses only history available by the displayed point.
    measured_so_far = pd.concat(
        [initial, previous, current],
        ignore_index=True,
    )
    best_observed_index = (
        measured_so_far[target].idxmax()
        if maximize
        else measured_so_far[target].idxmin()
    )
    best_observed = measured_so_far.loc[[best_observed_index]]
    best_observed_xy = project_measurements(best_observed)[0]
    best_observed_value = float(best_observed[target].iloc[0])

    optimum_id = (
        pool[target].idxmax()
        if maximize
        else pool[target].idxmin()
    )
    optimum_xy = pc.loc[optimum_id].to_numpy()
    optimum_value = float(pool.loc[optimum_id, target])

    # Keep the pool outcome color scale and axis limits identical everywhere.
    norm = Normalize(vmin=float(y.min()), vmax=float(y.max()))
    cmap = "viridis"

    all_xy = np.vstack([
        coordinates,
        initial_xy,
        previous_xy,
        current_xy,
    ])
    low, high = all_xy.min(axis=0), all_xy.max(axis=0)
    padding = np.maximum(high - low, 1e-9) * 0.06
    limits = (low - padding, high + padding)

    variance = pca.explained_variance_ratio_ * 100
    label = f"Round {int(round_id)}"
    stem = f"pca_round_{int(round_id):03d}"

    if stage_id is not None:
        label += f", stage {int(stage_id)}"
        stem += f"_stage_{int(stage_id):03d}"

    def background(ax):
        """Draw the same retrospective landscape in every panel."""
        scatter = ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=y,
            cmap=cmap,
            norm=norm,
            s=12,
            alpha=0.35,
            edgecolors="none",
            rasterized=True,
            zorder=1,
        )
        ax.set(
            xlabel=f"PC1 ({variance[0]:.1f}% variance)",
            ylabel=f"PC2 ({variance[1]:.1f}% variance)",
            xlim=(limits[0][0], limits[1][0]),
            ylim=(limits[0][1], limits[1][1]),
        )
        ax.grid(alpha=0.15)
        ax.set_axisbelow(True)
        return scatter

    def optimum_marker(ax, size=250):
        ax.scatter(
            *optimum_xy,
            marker="*",
            s=size,
            color="gold",
            edgecolors="black",
            linewidths=1.1,
            label="Global pool optimum",
            zorder=8,
        )

    output = Path(context.directory)

    # ---------------------------------------------------------
    # FIGURE 1: facility overview
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 8), constrained_layout=True)
    scatter = background(ax)

    if len(initial_xy):
        ax.scatter(
            initial_xy[:, 0],
            initial_xy[:, 1],
            marker="s",
            s=48,
            color="lightgray",
            edgecolors="dimgray",
            linewidths=0.8,
            label=f"Initial measurements ({len(initial)})",
            zorder=3,
        )

    if len(previous_xy):
        ax.scatter(
            previous_xy[:, 0],
            previous_xy[:, 1],
            s=45,
            color="#0072B2",
            edgecolors="white",
            linewidths=0.5,
            label=f"Earlier acquisitions ({len(previous)})",
            zorder=4,
        )

    ax.scatter(
        current_xy[:, 0],
        current_xy[:, 1],
        c=current[target].to_numpy(float),
        cmap=cmap,
        norm=norm,
        s=120,
        edgecolors="black",
        linewidths=1.5,
        label=f"Displayed selections ({len(current)})",
        zorder=5,
    )

    optimum_marker(ax, size=340)

    # An open diamond keeps the star visible if both markers coincide.
    ax.scatter(
        *best_observed_xy,
        marker="D",
        s=190,
        facecolors="none",
        edgecolors="#D55E00",
        linewidths=2.2,
        label="Best measured so far",
        zorder=9,
    )

    ax.annotate(
        f"Pool optimum: {optimum_value:.4g}",
        xy=optimum_xy,
        xytext=(12, 16),
        textcoords="offset points",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
        arrowprops=dict(arrowstyle="-", color="gray"),
        zorder=10,
    )

    ax.set_title(
        f"Facility selections — {label}\n"
        f"Best measured: {best_observed_value:.4g} | "
        f"Pool optimum: {optimum_value:.4g}",
        fontsize=14,
    )
    ax.legend(loc="best", fontsize=9, framealpha=0.95)
    fig.colorbar(
        scatter,
        ax=ax,
        label=f"{target}: true pool outcome (posthoc)",
        shrink=0.85,
    )
    fig.savefig(output / f"{stem}_overview.png", dpi=dpi)
    plt.close(fig)

    # ---------------------------------------------------------
    # FIGURE 2: stakeholder preferences versus facility choices
    # ---------------------------------------------------------
    round_submissions = submissions[
        submissions["round_id"] == round_id
    ].copy()

    agent_ids = list(dict.fromkeys(
        [str(a["agent_id"]) for a in cfg.get("agents", [])]
        + round_submissions["agent_id"].astype(str).tolist()
    ))

    if not agent_ids:
        raise ValueError("No stakeholders are available to plot.")

    ncols = min(3, len(agent_ids))
    nrows = ceil(len(agent_ids) / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6 * ncols, 5 * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    proposal_coordinates = []

    for ax, agent_id in zip(axes.flat, agent_ids):
        scatter = background(ax)

        proposals = round_submissions[
            round_submissions["agent_id"].astype(str) == agent_id
        ]

        if stage_id is not None:
            proposals = proposals[
                proposals["stage_id"] <= stage_id
            ]

        if not proposals.empty:
            # No refresh: this is the original round submission.
            # With refresh: use the latest snapshot available at this stage.
            proposals = proposals[
                proposals["stage_id"] == proposals["stage_id"].max()
            ]
            proposals = (
                proposals
                .sort_values("score", ascending=False, kind="stable")
                .drop_duplicates("candidate_id")
                .copy()
            )
            if top_k is not None:
                proposals = proposals.head(top_k)

        ids = proposals["candidate_id"].astype(str).tolist()
        unknown = set(ids) - set(pc.index)
        if unknown:
            raise ValueError(
                f"Proposal IDs missing from pool: {sorted(unknown)[:5]}"
            )

        proposed_xy = pc.loc[ids].to_numpy()

        if len(proposed_xy):
            ax.scatter(
                proposed_xy[:, 0],
                proposed_xy[:, 1],
                marker="^",
                s=90,
                color="#CC79A7",
                edgecolors="white",
                linewidths=0.6,
                label=f"Preferred proposals ({len(ids)})",
                zorder=4,
            )

            proposal_coordinates.append(pd.DataFrame({
                "agent_id": agent_id,
                "candidate_id": ids,
                "rank": np.arange(1, len(ids) + 1),
                "PC1": proposed_xy[:, 0],
                "PC2": proposed_xy[:, 1],
            }))

        # Open circles allow proposal markers to remain visible underneath.
        ax.scatter(
            current_xy[:, 0],
            current_xy[:, 1],
            s=130,
            facecolors="none",
            edgecolors="black",
            linewidths=1.5,
            label="Facility selections",
            zorder=6,
        )

        optimum_marker(ax)

        # Exact overlap supplements the visual comparison.
        selected_ids = set(current["candidate_id"].astype(str))
        overlap = len(set(ids) & selected_ids)

        ax.set_title(
            f"{agent_id}\n"
            f"{overlap} of {len(current)} selections in displayed shortlist",
            fontsize=12,
        )
        ax.legend(loc="best", fontsize=8, framealpha=0.9)

    for ax in list(axes.flat)[len(agent_ids):]:
        ax.set_visible(False)

    fig.suptitle(
        f"Stakeholder preferences and facility selections — {label}",
        fontsize=16,
    )
    fig.colorbar(
        scatter,
        ax=list(axes.flat)[:len(agent_ids)],
        label=f"{target}: true pool outcome (posthoc)",
        shrink=0.8,
        pad=0.02,
    )
    fig.savefig(output / f"{stem}_stakeholders.png", dpi=dpi)
    plt.close(fig)

    # Tables are saved through the existing analysis registry executor.
    coordinate_table = pc.reset_index()
    coordinate_table[target] = y
    coordinate_table["selected_in_display"] = (
        coordinate_table["candidate_id"].isin(
            current["candidate_id"].astype(str)
        )
    )

    tables = {f"{stem}_coordinates": coordinate_table}

    if proposal_coordinates:
        tables[f"{stem}_proposal_coordinates"] = pd.concat(
            proposal_coordinates,
            ignore_index=True,
        )

    return MetricResult(
        tables=tables,
        summary={
            "pca_plot_round": int(round_id),
            "pca_plot_stage": (
                int(stage_id) if stage_id is not None else None
            ),
            "pca_explained_variance_percent": [
                float(v) for v in variance
            ],
        },
    )



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_directory')
    parser.add_argument('--metrics', nargs='*', default=None,
                        help='Optional metric-name override; no names disables metrics.')
    args = parser.parse_args()
    print(analyze_run(args.run_directory, metrics=args.metrics))
