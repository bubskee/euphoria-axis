"""Shared, interruptible infrastructure for the frozen 27B battery."""
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PROTOCOL = REPO / 'doc/protocols/gen27b'
PAPER = REPO / 'runs/qwen35-4b-posttrained-s2/paper_protocol'
STORE = Path(os.environ.get('GEN27B_STORE', '/workspace/gen27b'))
LOCAL = Path(os.environ.get('GEN27B_LOCAL', '/root/gen27b'))
SOURCE = LOCAL / 'source'
STOP = False


class Interrupted(RuntimeError):
    pass


def request_stop(signum, frame):
    global STOP
    STOP = True
    print(f'Signal {signum}: finishing the current atomic unit, then stopping.', flush=True)


def install_signals():
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)


def check_stop():
    if STOP or (STORE / 'PAUSE').exists():
        raise Interrupted('Paused safely; completed units are saved. Remove PAUSE and resume the same command.')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(16 << 20), b''):
            h.update(part)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def progress(stage, **extra):
    record = {'stage': stage, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **extra}
    atomic_json(STORE / 'current.json', record)
    print(json.dumps(record), flush=True)


class Journal:
    """One fsynced record per completed unit; repair only a torn final line."""
    def __init__(self, path, identity, key_fields):
        self.path, self.identity, self.key_fields = Path(path), identity, key_fields
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records = {}
        if self.path.exists():
            with self.path.open('r+b') as f:
                while True:
                    start = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        if f.read(1):
                            raise ValueError(f'Corruption before final line: {self.path}')
                        f.truncate(start)
                        break
                    if not line.endswith(b'\n'):
                        # Even a valid JSON object needs a delimiter before appending.
                        f.seek(0, 2)
                        f.write(b'\n')
                    assert all(row.get(k) == v for k, v in identity.items()), f'Identity mismatch: {self.path}'
                    key = self.key(row)
                    assert key not in self.records, f'Duplicate completed unit {key}'
                    self.records[key] = row

    def key(self, row):
        return tuple(row[k] for k in self.key_fields)

    def contains(self, **row):
        return self.key(row) in self.records

    def append(self, row):
        row = {**row, **self.identity}
        key = self.key(row)
        assert key not in self.records
        with self.path.open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
            f.flush()
            os.fsync(f.fileno())
        self.records[key] = row


def read_spec(slug):
    path = PROTOCOL / f'{slug}.json'
    spec = json.loads(path.read_text())
    assert spec['status'] == 'frozen-before-27b-data'
    for rel, digest in spec['repo_inputs'].items():
        assert sha(REPO / rel) == digest, rel
    return spec, sha(path)


def bootstrap_ci(values, n_boot=20000, seed=20260925):
    values = np.asarray(values, dtype=float)
    assert values.ndim == 1 and len(values) and np.isfinite(values).all()
    idx = np.random.default_rng(seed).integers(0, len(values), size=(n_boot, len(values)))
    lo, hi = np.quantile(values[idx].mean(1), [.025, .975])
    return [float(values.mean()), float(lo), float(hi)]


def fit_direction(pain, control):
    """The 4B Gram-matrix control-PCA recipe, fitted using supplied rows only."""
    import torch
    raw = pain.mean(0) - control.mean(0)
    centered = control - control.mean(0)
    eig, vec = torch.linalg.eigh(centered @ centered.T)
    order = torch.argsort(eig, descending=True)
    eig, vec = eig[order].clamp_min(0), vec[:, order]
    assert eig.sum() > 0
    k = int(torch.searchsorted(eig.cumsum(0) / eig.sum(), .5)) + 1
    basis = centered.T @ vec[:, :k]
    basis /= eig[:k].clamp_min(1e-8).sqrt()[None, :]
    denoised = raw - basis @ (basis.T @ raw)
    assert torch.isfinite(denoised).all() and denoised.norm() > 0
    return raw, denoised, k


def unit(vector):
    return (vector.double() / vector.double().norm()).float()


def execution_metadata(spec, spec_sha):
    import platform
    import torch
    import transformers
    code = {p.name: sha(p) for p in Path(__file__).parent.glob('gen27b*') if p.is_file()}
    return {'model_id': spec['model_id'], 'model_revision': spec['model_revision'],
            'spec_sha256': spec_sha, 'torch': torch.__version__, 'transformers': transformers.__version__,
            'python': platform.python_version(), 'gpu': torch.cuda.get_device_name(0),
            'code_sha256': code, 'git_revision': subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
            'environment_sha256': sha(STORE / 'environment.lock.txt')}


def complete(directory, module, paths, spec_sha):
    atomic_json(Path(directory) / f'{module}.complete.json',
                {'spec_sha256': spec_sha, 'artifacts': {str(Path(p).relative_to(directory)): sha(p) for p in paths}})


def is_complete(directory, module, spec_sha):
    path = Path(directory) / f'{module}.complete.json'
    if not path.exists():
        return False
    card = json.loads(path.read_text())
    assert card['spec_sha256'] == spec_sha
    assert all(sha(Path(directory) / p) == h for p, h in card['artifacts'].items())
    return True
