"""Construct and launch an experiment; this is the entry point to read first.

Connections: config -> pool + agents + facility + storage -> Simulation.run()
-> analysis.analyze_run(). The constructors choose implementations; the loop in
simulation.py does not branch on individual model families.

Run from this directory: python main.py --demo
For your own data: python main.py --config experiment.example.json
"""
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
# Shared contracts/configuration are imported first; other modules consume these objects.
from config import ExperimentConfig, FacilityConfig, AgentConfig, metric_configs, load_extensions
from agents import build_agents
from pool import CSVPool, prepare_pool
from facility_policies import FACILITY_POLICIES
# Simulation coordinates components but does not construct scientific models.
from simulation import Simulation
from storage import RunStorage
from analysis import analyze_run

def load_config(path):
    """Read JSON and build typed configuration containers.
    
    Relative paths are resolved from the JSON file location, not the terminal
    working directory. AgentConfig and FacilityConfig are nested settings.
    """
    path = Path(path).resolve()
    # Load settings as plain Python values before constructing dataclasses.
    raw = json.loads(path.read_text(encoding='utf-8'))
    # Relative data/output paths are resolved against the config file directory.
    for key in ['pool_path', 'output_directory']:
        if not Path(raw[key]).is_absolute():
            raw[key] = str(path.parent/raw[key])
    raw['initial_paths'] = tuple(str(path.parent/p) if not Path(p).is_absolute() else p for p in raw.get('initial_paths', []))
    raw['features'] = tuple(raw['features'])
    # Construction choices now belong to agents.py's factory, not to main.py.
    raw['agents'] = [AgentConfig(**item) for item in raw['agents']]
    raw['base_directory'] = str(path.parent)
    if 'analysis_metrics' in raw:
        raw['analysis_metrics'] = metric_configs(raw['analysis_metrics'])
    raw['facility'] = FacilityConfig(**raw.get('facility', {}))
    return ExperimentConfig(**raw)

def run(config):
    """Wire the components together, execute the engine, then analyze saved data.
    
    Returns the run-directory Path. Used by the command-line entry point and tests.
    """
    # The full labels remain inside this object; agents will receive public snapshots.
    pool = CSVPool(config)
    # Extensions register their classes/functions before any factory lookup.
    load_extensions(config.extension_modules)
    agents = build_agents(config.agents, base_directory=config.base_directory)
    # This single registry lookup chooses the facility algorithm.
    policy = FACILITY_POLICIES[config.facility.policy](config.facility)
    # Inject all components into the same engine; run() returns its saved output path.
    path = Simulation(config, pool, agents, policy, RunStorage(config)).run()
    # Analyze saved records only after acquisition and measurement have completed.
    print('Analysis:', analyze_run(path))
    return path

def demo_config(base):
    """Create deterministic synthetic data and settings for a quick software check.
    
    Uses three lightweight agent implementations; every run gets a fresh output
    name. The CSVs are artificial and do not represent experimental validation.
    """
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    # Generate synthetic candidate parameters; there is no connection to the uploaded data.
    X = rng.uniform(0.2, 2.0, (45, 3))
    # Positive synthetic outcomes satisfy the notebook agents log-target requirement.
    y = np.exp(0.4*np.sin(X[:, 0]*3)+0.2*X[:, 1]-0.1*X[:, 2])
    f = pd.DataFrame(X, columns=['x1', 'x2', 'x3'])
    f.insert(0, 'candidate_id', [f'candidate_{i:03}' for i in range(len(f))])
    f['y'] = y
    f.to_csv(base/'pool.csv', index=False)
    # These candidates are already measured and therefore excluded from future selection.
    f.iloc[:12].to_csv(base/'initial.csv', index=False)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    return ExperimentConfig(str(base/'pool.csv'), ('x1', 'x2', 'x3'), (str(base/'initial.csv'),), str(base/f'run_{stamp}'), agents=[AgentConfig('gp_upper', model='gp', parameters={'backend': 'sklearn'}), AgentConfig('chebyshev_upper', model='chebyshev', parameters={'degree': 3}), AgentConfig('bijector_upper', model='bijector', parameters={'n_splits': 5})], rounds=2, batch_size=6, facility=FacilityConfig(calibration_checkpoints=(3,), pca_components=3))

# Only execute command-line parsing when run directly, not when imported by tests.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    # Require exactly one choice: a synthetic demo OR the user-provided configuration.
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--demo', action='store_true')
    group.add_argument('--config')
    parser.add_argument('--demo-directory', default='demo_data')
    args = parser.parse_args()
    run(demo_config(args.demo_directory) if args.demo else load_config(args.config))
