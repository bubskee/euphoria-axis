"""Storage-only integrity primitives. Never computes a model or lens result."""
import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encode(row):
    # Exactly the frozen Journal.append serialization.
    return (json.dumps(row, allow_nan=False) + '\n').encode()


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read_exact(path, expected, attempts=8):
    """Never accept a stale/truncated read, even when its own hash is valid."""
    observed = None
    for attempt in range(attempts):
        try:
            observed = Path(path).read_bytes()
        except FileNotFoundError:
            observed = None
        if observed == expected:
            return observed
        if attempt + 1 < attempts:
            time.sleep(min(.1 * 2 ** attempt, 2))
    raise IOError(f'Read-back mismatch at {path}: wanted {len(expected)} bytes, '
                  f'observed {None if observed is None else len(observed)}')


def seal(directory, data, suffix='.jsonl'):
    """A new immutable name avoids mutable-object generation/cache ambiguity."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (digest(data) + '-' + uuid.uuid4().hex + suffix)
    pending = path.with_name(path.name + '.pending')
    with pending.open('xb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    read_exact(pending, data)
    os.replace(pending, path)
    fsync_directory(directory)
    read_exact(path, data)
    return path


def read_sealed(path, attempts=8):
    """Committed object names bind bytes; short immutable views are not rows."""
    path = Path(path)
    match = re.search(r'[0-9a-f]{64}', path.name)
    if not match:
        raise ValueError('Sealed observation has no content hash: ' + str(path))
    wanted = match.group()
    for attempt in range(attempts):
        data = path.read_bytes()
        if digest(data) == wanted:
            return data
        if attempt + 1 < attempts:
            time.sleep(min(.1 * 2 ** attempt, 2))
    raise IOError('Immutable observation hash mismatch: ' + str(path))


def publish(path, data, sealed_directory=None):
    """Publish only read-back-verified immutable bytes, then verify the alias."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sealed_directory = Path(sealed_directory or path.parent / '.sealed-storage')
    immutable = seal(sealed_directory, data)
    for attempt in range(3):
        candidate = seal(path.parent / '.storage-installs', data, '.tmp')
        os.replace(candidate, path)
        fsync_directory(path.parent)
        try:
            read_exact(path, data)
            return immutable
        except IOError:
            if attempt == 2:
                raise


class RowSet:
    """Original serialized rows, checked by identity, key and exact bytes."""
    def __init__(self, identity, key_fields):
        self.identity = identity
        self.key_fields = key_fields
        self.records = {}
        self.raw = {}
        self.sources = {}

    def key(self, row):
        return tuple(row[k] for k in self.key_fields)

    def add(self, data, source):
        if data and not data.endswith(b'\n'):
            raise ValueError(f'Unterminated observation: {source}')
        within = set()
        for line in data.splitlines(keepends=True):
            row = json.loads(line)
            if not all(row.get(k) == v for k, v in self.identity.items()):
                raise ValueError(f'Identity mismatch: {source}')
            key = self.key(row)
            if key in within:
                raise ValueError(f'Duplicate within {source}: {key}')
            within.add(key)
            if key in self.raw and self.raw[key] != line:
                raise ValueError(f'Conflicting serialized row: {key} from {source}')
            self.raw[key] = line
            self.records[key] = row
            self.sources.setdefault(key, []).append(str(source))

    def missing(self, expected):
        return [key for key in expected if key not in self.records]

    def materialize(self, expected):
        expected = list(expected)
        missing = self.missing(expected)
        extra = set(self.records) - set(expected)
        if missing or extra or len(set(expected)) != len(expected):
            raise ValueError(f'Incomplete row set: {len(missing)} missing, {len(extra)} extra')
        return b''.join(self.raw[key] for key in expected)


def lens_keys(spec):
    conditions = ['baseline'] + [f'{name}_{co:+.1f}'
        for name in ['S2', 'S1', *[f'random_{s}' for s in spec['random_seeds']]]
        for co in [-.5, .5]]
    return [(condition, prompt, layer, label)
        for condition in conditions for prompt in range(spec['m6']['prompts'])
        for layer in range(spec['n_layers'] - 1) for label in spec['lenses']]


class CheckedReplay:
    """Allow missing-row recovery only after exact surviving-row comparisons."""
    def __init__(self, rows, expected, cases, persist):
        self.rows = rows
        self.expected = set(expected)
        self.order = list(expected)
        self.cases = set(cases)
        self.persist = persist
        self.verified = set()
        self.added = []

    def contains(self, **row):
        key = self.rows.key(row)
        return key in self.rows.records and key[:2] not in self.cases

    def append(self, row):
        row = {**row, **self.rows.identity}
        key, data = self.rows.key(row), encode(row)
        if key not in self.expected or key[:2] not in self.cases or key in self.verified:
            raise ValueError('Unexpected or duplicate replay key: ' + str(key))
        if key in self.rows.raw:
            if self.rows.raw[key] != data:
                raise ValueError('Frozen replay disagrees with an original rank row: ' + str(key))
        else:
            self.persist(data)
            self.rows.add(data, 'exact frozen missing-row replay')
            self.added.append(key)
        self.verified.add(key)

    def finish(self):
        required = {key for key in self.expected if key[:2] in self.cases}
        if self.verified != required:
            raise ValueError('Not every surviving row in each replayed prompt was checked')
        return self.rows.materialize(self.order)


class ImmutableJournal:
    """Frozen Journal API with per-row persistent immutable WAL and POSIX mirror.

    Used only by newly launched processes. It never hot-patches a running worker.
    The live 3.6 worker retains its already-loaded original storage implementation.
    """
    registry = {}
    expected_keys = {}
    root_mirror = Path('/root/gen27b/storage-recovery/mirror')

    def __init__(self, path, identity, key_fields):
        self.path = Path(path)
        self.identity, self.key_fields = identity, key_fields
        self.rows = RowSet(identity, key_fields)
        self.wal = self.path.parent / (self.path.stem + '.immutable-wal')
        mirror_key = digest(str(self.path).encode())
        self.mirror = self.root_mirror / mirror_key
        if self.path.exists():
            self.rows.add(self.path.read_bytes(), str(self.path))
        # Multiple exact copies are mirrors, never conflicting/duplicate science.
        for base in (self.wal, self.mirror):
            if base.exists():
                for p in sorted(base.glob('*/*.jsonl')):
                    self.rows.add(read_sealed(p), str(p))
        self.records = self.rows.records
        self.order = self.expected_keys.get(self.path.name)
        if self.order is not None:
            if set(self.records) - set(self.order):
                raise ValueError('Unexpected immutable journal key')
        self.registry[str(self.path)] = self
        self.flush()

    def key(self, row):
        return self.rows.key(row)

    def contains(self, **row):
        return self.key(row) in self.records

    def append(self, row):
        row = {**row, **self.identity}
        key = self.key(row)
        if key in self.records:
            raise ValueError(f'Duplicate completed unit: {key}')
        data = encode(row)
        fanout = digest(json.dumps(key).encode())[:2]
        # Root mirror is additional protection, never the sole acknowledged copy.
        seal(self.mirror / fanout, data)
        seal(self.wal / fanout, data)
        self.rows.add(data, 'acknowledged immutable WAL')
        self.flush()

    def flush(self):
        if self.order is not None:
            ordered = {key: self.records[key] for key in self.order if key in self.records}
            self.records.clear()
            self.records.update(ordered)
        data = b''.join(self.rows.raw[key] for key in self.records)
        publish(self.path, data, self.wal / 'canonical-objects')
        # Exact bytes, not merely a self-consistent hash of a short read.
        read_exact(self.path, data)
        return data

    def require(self, expected):
        wanted = self.rows.materialize(expected)
        # Reorder only when the frozen expected iteration order differs on resume.
        publish(self.path, wanted, self.wal / 'canonical-objects')
        read_exact(self.path, wanted)
        return digest(wanted)
