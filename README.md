## Run the included example

Open a terminal in this extracted directory (Python 3.11+):

```bash
python -m pip install -r requirements.txt
python main.py --demo
python -m unittest discover -s tests -v
```

The demo uses the scikit-learn GP backend, Chebyshev Bayesian regression, and
bijector polynomial ensemble. It runs two rounds and writes a new timestamped
run directory under `demo_data`. It does not need PyTorch or XGBoost.

For five model families (stakeholders):

```bash
python -m pip install -r requirements-full.txt
python main.py --config experiment.example.json
```

Before the second command, prepare the CSVs below and edit the example JSON.
The example selects 26 new experiments from one submission round. Increase
`rounds` to retrain agents and repeat the process with shared observations.
The Lorenz ensemble defaults to 9,000 epochs and five models, as in the notebook,
so running all agents may take substantial time. Dependencies are minimum
constraints, not a fully pinned environment; each run records installed versions.

## Files and extension points

| File | Responsibility |
|---|---|
| main.py | Construct components, load JSON settings, run demo or experiment |
| config.py | Settings, Context, Prediction, Selection |
| pool.py | CSV validation, one-time preparation, label revelation |
| agents.py | Five surrogate families (stakeholders), target transforms, proposal rules, CSV replay |
| submission_processing.py | Column aliases, ID validation, score normalization |
| facility_policies.py | Precision aggregation, EI, PCA/RBF Vendi, calibration, random baseline |
| simulation.py | Information timing, stages, feedback through shared observations |
| storage.py | CSV recording, configuration, dependency and source/input hashes |
| analysis.py | Best-observed curves, proposal statistics, final-run comparisons |

## Prepare your existing data

Create a single `data/pool.csv` with explicit stable IDs, feature columns, and `y`.
Create `data/initial_observations.csv` containing everything already measured,
including the original first batch and iterations 1–4 when reproducing iteration 5.
Use the actual feature column names, in an explicit order. Do not include exported
CSV index columns as features.

You can run the following in a small temporary script or notebook in this directory:

```python
from pool import prepare_pool, prepare_initial_history

features = ['PARAMETER_1', 'PARAMETER_2', 'PARAMETER_3']
prepare_pool(
    search_path='First_SearchSpace.csv',
    labeled_path='CleanedData.csv',
    output_path='data/pool.csv',
    features=features,
    target='y',
)
prepare_initial_history(
    batch_paths=[
        'First_DataBatch.csv',
        'iteration1_newdatacollected.csv',
        'iteration2_newdatacollectedCORRECTECTED.csv',
        'iteration3_newdatacollected.csv',
        'iteration4_newdatacollected.csv',
    ],
    prepared_pool_path='data/pool.csv',
    output_path='data/initial_observations.csv',
    features=features,
)
```

The first function joins outcomes by exact parameter values and assigns IDs before
any filtering. The second matches historical observations to those IDs. Initial
points outside the search space receive separate IDs. Ambiguous duplicates and
unmatched outcomes cause errors, rather than silent selection changes. For
floating-point values that do not match exactly, define a domain-specific matching
rule before using these functions. Replicated noisy measurements need a separate
replicate schema; this version assumes one deterministic outcome per candidate.

Then edit `experiment.example.json`:

- Set `features` to the exact names used above.
- Set the data paths and choose a NEW output directory for every run.
- Set agents, seeds, batch size, and rounds.
- Relative paths are resolved against the JSON's directory.
- Targets must be strictly positive for the notebook's log-transform agents.
- At least four initial observations are required; very small sets may still be
  unsuitable for individual model methods.

## Agent behavior

Every ModelAgent trains from the current public observations, then scores all
available candidates. Optional `proposal_limit` retains a ranked shortlist.
The same history is provided to every agent. By default, each agent retrains once
per round; its predictions remain fixed during that round, matching  original
CSV-based procedure. Facility calibration may still filter prediction contributors.

Supported model names: `gp`, `bijector`, `xgb`, `chebyshev`, `lorenz`.
Supported proposal rules: `upper`, `uncertainty`, `ei`.
An agent's `parameters` dictionary is passed to its surrogate constructor.

For the original GP training use `backend: gpytorch`; the optional `sklearn`
backend is an alternative optimizer/model implementation used by the demo.
The GP returns latent uncertainty, matching the notebook's use of `model(x)`;
it does not add likelihood noise to the reported uncertainty.

To add another model, implement `fit(X, y)` and `predict(X)` in `agents.py`, then:

```python
MODELS['my_model'] = MySurrogate
```

The generic agent fits input standardization and log-target standardization using
measured data, then converts predictive Gaussian moments to original-scale
log-normal moments. Custom non-Gaussian agents can implement `propose(context)`
directly and return the same submission table. The current facility requires
finite original-scale means and standard deviations for a common scalar outcome.
It moment-matches non-Gaussian predictions; it does not perform exact log-normal EI.

To add a proposal rule, register a function with the same signature as `ei_score`
in `RULES`. To add more agents, add JSON entries with unique agent IDs. Separate
instances own their own fitted models. Agent seeds are explicit; changing the
experiment seed alone does not overwrite per-agent seeds.

## Measurement and calibration timing

- `sequential`: reveal one outcome at a time; recompute facility quality each step.
- `staged`: select batches ending at the calibration checkpoints, e.g. 10, 10, 6.
- `batch`: select all 26 before revealing any outcomes. Mid-batch calibration
  cannot occur in this mode.

`refresh_agents_each_stage: false` keeps stakeholder predictions fixed within a
round. Set it to true to regenerate submissions after each measurement stage.
In sequential mode, this retrains every agent after every measured sample.
Pre-outcome predictions are retained for calibration, avoiding in-sample rechecks.

The facility uses equal mean normalized stakeholder score plus normalized EI,
weighted 0.5/0.5 by default. It greedily maximizes mean selected RAW quality times
Vendi diversity. The diversity reference includes all measured feature vectors,
including the initial batch, plus the batch under construction. PCA uses only
features, never hidden targets; fitting on candidate features is transductive.
By default, no feature scaling is used before PCA, matching the notebook. Set
`scale_features` to true deliberately if desired. Full eigendecomposition inside
selection is intended for initial experiments; large pools/histories may need
kernel caching or approximations.

At checkpoints 10 and 20 the policy computes Gaussian interval coverage error for
each stakeholder using the outcomes revealed so far in the current round. It
min–max scales these errors and retains prediction contributors with relative
error <= 0.5. Equal errors retain all models. This is a RELATIVE model filter, not
uncertainty recalibration, and its result changes with agent composition. Agents
without calibration evidence remain eligible. Mean stakeholder scores continue
to use all valid submissions, matching the notebook's score aggregation design.

For partial shortlists, candidates with no eligible prediction contributor are
excluded. Random selection samples the union of valid proposed candidates, not
the entire pool. With full-pool submissions these supports coincide.

## Intentional corrections versus the notebooks

1. Preserve canonical candidate IDs, never recreate filtered positions with arange.
2. Keep selected raw quality separate from composite quality–Vendi scores.
3. Compute eligibility once per checkpoint; do not accumulate repeated model
   columns across candidate rows, which artificially shrinks uncertainty.
4. Handle constant min–max arrays and all-zero quality without undefined values.
5. Normalize EI consistently whenever quality is recomputed.
6. Use real predictive standard deviations, not interval widths mislabeled as std.
   Original-scale means/std are moment-matched from latent Gaussian predictions;
   exported legacy interval-width CSVs must be converted separately.
7. Preserve the optimistic exp(m+s) upper score for default maximization agents.
   This is an upper log-normal bound, not mean+std on the original scale.
8. Polynomial uncertainty uses ensemble standard deviation, not the 5–95% width.
   Reciprocal inputs are guarded near zero and lstsq handles singular designs.
9. XGBoost feature filtering falls back to all original features if none exceeds
   the dummy feature. Random splits and models have explicit seeds.
10. Neural input size follows the pool rather than being fixed to 11; the Lorenz
    z initialization uses the third input instead of overwriting it from x.
    Ensemble members train independently rather than vectorizing one joint loss;
    uncertainty uses population std consistently across ensemble methods.
11. Facility labels are accessible only through pool.evaluate in the engine.
    Predictions can remain fixed while the facility's EI incumbent is updated
    after each sequential outcome, unlike the notebook's checkpoint-only update.
12. All initial observations are included in the diversity reference. Therefore,
    exact old selections are not promised even after supplying original data.

The manual central-interval MACE implementation uses an evenly spaced grid over
[0,1], with five levels by default. It avoids an uncertainty_toolbox dependency.
It was not numerically cross-validated against your installed toolkit version.
Precision aggregation retains the notebook's independent unbiased Gaussian
assumption; shared-data model correlations can make it overconfident.

## Saved records and analysis

Every run includes configuration, metadata, initial observations, raw submissions
per agent, standardized submissions, validation reports (if any), selected IDs and
scores, revealed outcomes, and calibration history (if checkpoints were reached).
A COMPLETE marker is written only on successful completion. Existing output
folders are never overwritten.

```bash
python analysis.py results/experiment_001
```

Best-observed curves respect stage/batch visibility. Proposal statistics count
unique per-round agent/candidate submissions and matched selections. These are
NOT an equitable allocation score when every agent submits the entire pool.
For access fairness, configure proposal quotas/shortlists and define shared-credit
rules first. Benefit fairness, influence/Shapley, RMSE-AUC, and waiting-time
metrics are extension points; they are not claimed to be implemented here.

For a random baseline, copy the experiment config, keep the same data, agent
settings, measurement timing, and budget, set `facility.policy` to `random`, and
choose a new output directory. Then compare separately executed runs:

```python
from analysis import compare_runs
compare_runs(['results/quality_run', 'results/random_run'], 'comparison.csv')
```

This comparison exports final best outcomes and measurement counts. Statistical
claims require repeated seeds. Adaptive baseline runs regenerate proposals from
their own histories, rather than reusing the selected policy's future CSVs.

## Replay existing CSV submissions

Configure an agent with:

```json
{"agent_id": "external_A", "model": "csv",
 "parameters": {"path_template": "submissions/round_{round}/external_A.csv"}}
```

Files need candidate_id, prediction, uncertainty (true std), and score. Aliases
Index/pool_id and std are accepted by submission processing, but aliasing does
not repair incorrect original IDs. Old notebook exports that renumbered filtered
rows need the exact candidate order used to generate each file; do not guess.
Legacy interval widths also need their model-specific statistical conversion.
CSV replay cannot automatically adapt to changed facility allocations.

## Validation in this delivery

Unit tests and synthetic end-to-end tests cover math, stable IDs, constant score
handling, calibration evidence size, hidden target columns, sequential/staged/batch
execution, random selection and overwrite protection. GP (scikit-learn backend),
Chebyshev and bijector agents were exercised. PyTorch, GPyTorch and XGBoost were
unavailable in the execution environment: their wrappers were syntax-checked,
but training was NOT executed. No original-data numerical reproduction was possible.
