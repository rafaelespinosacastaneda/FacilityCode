"""Coordinate the experiment and control when information becomes available.

main.py injects the pool, agents, facility, and recorder. The engine repeatedly:
constructs a public Context -> collects and validates submissions -> asks the
facility for IDs -> reveals those IDs through pool.evaluate() -> records results
and updates public observations. New agent families do not require loop changes.

A round is one stakeholder-submission cycle. A stage is one group of measurements
whose outcomes become available together. Sequential mode has stages of size one;
staged mode splits at checkpoints; batch mode reveals an entire batch at once.
"""
import pandas as pd
from config import Context
from submission_processing import process_submissions

class Simulation:
    """Orchestrator depending only on common agent and facility interfaces.
    """
    def __init__(self, config, pool, agents, facility, storage):
        """Inject configured components; no model-specific construction happens here.
        """
        self.config, self.pool, self.agents = config, pool, agents
        self.facility, self.storage = facility, storage

    def run(self):
        """Run rounds/stages, record each event, and return the completed run directory.
        
        Outcomes become public only after selection. Updating the shared observed
        table is how these model agents receive feedback on their next propose() call.
        """
        cfg = self.config
        if cfg.measurement_mode not in {'sequential', 'staged', 'batch'}:
            raise ValueError('Unknown measurement mode.')
        if cfg.direction not in {'maximize', 'minimize'} or cfg.batch_size < 1:
            raise ValueError('Invalid direction or batch size.')
        if len({a.agent_id for a in self.agents}) != len(self.agents):
            raise ValueError('Agent IDs must be unique.')
        # Initialize the public history once; this replaces manually named batch variables.
        observed = self.pool.initial_observations()
        self.storage.append('initial_observations', observed)
        # Start a fresh proposal round, potentially training every model on a larger history.
        for round_id in range(cfg.rounds):
            selected_in_round = []
            # The closure reads the latest observed table after each stage updates it.
            def context():
                """Build a fresh public view using the latest revealed observations.
                
                This nested helper reads the enclosing round state; hidden labels stay in pool.
                """
                return Context(round_id, self.pool.candidates(observed.candidate_id), observed.copy(deep=True), cfg.features, cfg.target, cfg.direction, tuple(selected_in_round))
            ctx = context()
            if ctx.candidates.empty:
                break
            def collect(ctx, stage_id):
                # Each agent gets its own snapshot of the same public state.
                """Ask each agent for proposals and record both raw and validated submissions.
                
                Each agent receives a separate copy of the same public snapshot. Return the
                standardized table used by the facility; retain rejections for inspection.
                """
                # All agents see equal information, and copies isolate accidental in-place edits.
                frames = [a.propose(Context(ctx.round_id, ctx.candidates.copy(deep=True), ctx.observations.copy(deep=True), ctx.features, ctx.target, ctx.direction, ctx.selected_in_round)) for a in self.agents]
                for agent_index, f in enumerate(frames):
                    self.storage.append(f'raw_submissions_agent_{agent_index:03}', f, round_id=round_id, stage_id=stage_id)
                # Enforce the common submission schema before the facility can use the data.
                submissions, report = process_submissions(frames, ctx.candidates.candidate_id)
                self.storage.append('submissions', submissions, round_id=round_id, stage_id=stage_id)
                self.storage.append('validation_report', report, round_id=round_id, stage_id=stage_id)
                return submissions
            submissions = collect(ctx, 0)
            # Initialize round-specific PCA, issued predictions and calibration state.
            self.facility.begin_round(submissions, ctx)
            if cfg.measurement_mode == 'sequential':
                # Cumulative boundaries 1,2,... mean each stage reveals one measurement.
                boundaries = list(range(1, cfg.batch_size+1))
            elif cfg.measurement_mode == 'staged':
                # For checkpoints 10 and 20 with budget 26, stage sizes become 10,10,6.
                boundaries = sorted({c for c in cfg.facility.calibration_checkpoints if 0 < c < cfg.batch_size})+[cfg.batch_size]
            else:
                # A single batch stage cannot access any new outcomes until all its IDs are selected.
                boundaries = [cfg.batch_size]
            for stage_id, boundary in enumerate(boundaries):
                ctx = context()
                if ctx.candidates.empty:
                    break
                # Optionally rerun stakeholders within a round; default False preserves fixed submissions.
                if stage_id and cfg.refresh_agents_each_stage:
                    submissions = collect(ctx, stage_id)
                    self.facility.refresh(submissions, ctx)
                # Pass only public data; the return value is a decision, not an experimental outcome.
                decision = self.facility.select(submissions, ctx, boundary-len(selected_in_round))
                ids = decision.candidate_ids
                if not ids:
                    break
                # Validate IDs against BOTH remaining pool candidates and valid submitted proposals.
                allowed = set(ctx.candidates.candidate_id)&set(submissions.candidate_id)
                if len(ids) != len(set(ids)) or len(ids) > boundary-len(selected_in_round) or not set(ids).issubset(allowed):
                    raise ValueError('Facility returned invalid selection.')
                # No oracle access occurs inside agents or facility methods.
                # Information boundary: this is where previously hidden outcomes become available.
                measured = self.pool.evaluate(ids)
                selected_in_round.extend(ids)
                self.storage.append('selections', decision.diagnostics, round_id=round_id, stage_id=stage_id)
                self.storage.append('observations', measured, round_id=round_id, stage_id=stage_id)
                # Broadcast feedback by updating the shared history used in every subsequent agent context.
                observed = pd.concat([observed, measured], ignore_index=True)
                # Facility calibration can now compare past predictions against the newly revealed labels.
                self.facility.observe(measured, cfg.target)
                print(f'Round {round_id}, stage {stage_id}: measured {len(ids)}, total new {len(selected_in_round)}', flush=True)
            if not selected_in_round:
                break
        # Persist final calibration history and mark successful completion.
        return self.storage.complete(self.facility.calibration_log)
