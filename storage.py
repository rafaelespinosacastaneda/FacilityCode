"""Persist run evidence so analysis can be rerun without repeating experiments.

main.py constructs RunStorage and simulation.py calls append()/complete().
Configuration and metadata are written once; event tables are appended as stages
finish. analysis.py consumes these CSVs independently of the agents and facility.
A fresh directory is required to avoid mixing different experiments.
"""
from dataclasses import asdict
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import pandas as pd

class RunStorage:
    """Write experiment evidence into one new directory; never overwrite a run.
    """
    def __init__(self, config):
        """Create output directory and record settings, dependency versions and hashes.
        
        Hashing inputs and source files lets later analyses identify exact versions.
        """
        self.directory = Path(config.output_directory)
        # exist_ok=False intentionally prevents overwriting an earlier run.
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory/'config.json').write_text(json.dumps(asdict(config), indent=2), encoding='utf-8')
        # Track environment/input/source fingerprints so results can be traced to an implementation.
        metadata = {'python': platform.python_version(), 'packages': {}, 'input_hashes': {}, 'source_hashes': {}}
        for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'torch', 'gpytorch', 'xgboost']:
            try:
                metadata['packages'][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        # Hash source bytes instead of trusting a filename that could later be replaced.
        for name in [config.pool_path, *config.initial_paths]:
            metadata['input_hashes'][str(name)] = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        for path in Path(__file__).parent.glob('*.py'):
            metadata['source_hashes'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        (self.directory/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    def append(self, name, frame, **identifiers):
        """Append a nonempty DataFrame to an event CSV with round/stage identifiers.
        
        The first append writes column names; subsequent calls expect the same schema.
        Frames are copied so adding identifiers does not mutate policy/agent data.
        """
        if frame.empty:
            return
        # Adding round/stage columns must not modify the table used by other components.
        frame = frame.copy()
        for key, value in identifiers.items():
            frame[key] = value
        path = self.directory/f'{name}.csv'
        # Append one event block; write a header only when creating a new table.
        frame.to_csv(path, mode='a', header=not path.exists(), index=False)

    def complete(self, calibration_log):
        """Write calibration history and a completion marker; return the run path.
        """
        if calibration_log:
            pd.DataFrame(calibration_log).to_csv(self.directory/'calibration_history.csv', index=False)
        # Absence of this marker indicates that the run did not finish successfully.
        (self.directory/'COMPLETE').write_text('Run completed\n')
        return self.directory
