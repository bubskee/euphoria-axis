"""Storage-only lens journal adapter: durable small chunks, unchanged row math."""
import hashlib
import json
import os
import runpy
from pathlib import Path

from gen27b_common import atomic_json, sha


class ChunkedJournal:
    """Keep every row fsynced on the same volume without growing each upload.

    The original journal is snapshotted byte-for-byte. Chunks retain the exact
    original serialization. The canonical journal is assembled once at the
    original runner's completion boundary, and chunks remain for safe resume.
    """
    chunk_rows = 128

    def __init__(self, path, identity, key_fields):
        self.canonical = Path(path)
        self.identity, self.key_fields, self.records = identity, key_fields, {}
        self.directory = self.canonical.parent / (self.canonical.stem + '.durable-chunks')
        self.directory.mkdir(parents=True, exist_ok=True)
        self.prefix = self.directory / 'prefix.jsonl'
        self.meta = self.directory / 'storage.json'
        expected = {'identity': identity, 'key_fields': key_fields, 'chunk_rows': self.chunk_rows,
                    'adapter_sha256': sha(__file__)}
        if not self.meta.exists():
            data = self.canonical.read_bytes() if self.canonical.exists() else b''
            # A complete pause boundary must end with a whole newline-delimited row.
            assert not data or data.endswith(b'\n'), 'Legacy journal has an incomplete tail; preserve and review'
            if self.prefix.exists():
                assert self.prefix.read_bytes() == data, 'Interrupted snapshot differs from legacy journal'
            else:
                tmp = self.prefix.with_suffix('.tmp')
                with tmp.open('wb') as f:
                    f.write(data); f.flush(); os.fsync(f.fileno())
                tmp.replace(self.prefix)
            atomic_json(self.meta, {**expected, 'prefix_bytes': len(data),
                                    'prefix_sha256': hashlib.sha256(data).hexdigest()})
        meta = json.loads(self.meta.read_text())
        assert all(meta[k] == v for k, v in expected.items()), 'Storage resume identity changed'
        assert self.prefix.stat().st_size == meta['prefix_bytes'] and sha(self.prefix) == meta['prefix_sha256']
        self._read(self.prefix)
        self.prefix_records = len(self.records)
        chunks = sorted(self.directory.glob('chunk-*.jsonl'))
        assert [p.name for p in chunks] == [f'chunk-{i:06d}.jsonl' for i in range(len(chunks))]
        self.chunks = chunks
        self.active_rows = 0
        for i, chunk in enumerate(chunks):
            count = self._read(chunk, repair=(i == len(chunks)-1))
            assert count <= self.chunk_rows
            if i < len(chunks)-1:
                assert count == self.chunk_rows, 'Incomplete non-final chunk'
            self.active_rows = count

    def key(self, row):
        return tuple(row[k] for k in self.key_fields)

    def contains(self, **row):
        return self.key(row) in self.records

    def _read(self, path, repair=False):
        data = path.read_bytes()
        lines, offset, count = data.splitlines(keepends=True), 0, 0
        for i, line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if not repair or i != len(lines)-1:
                    raise
                with path.open('r+b') as f:
                    f.truncate(offset); f.flush(); os.fsync(f.fileno())
                break
            assert all(row.get(k) == v for k, v in self.identity.items()), f'Identity mismatch: {path}'
            key = self.key(row)
            assert key not in self.records, f'Duplicate completed unit {key}'
            self.records[key] = row
            count += 1
            offset += len(line)
            if not line.endswith(b'\n'):
                assert repair and i == len(lines)-1, 'Unterminated immutable prefix or non-final row'
                with path.open('ab') as f:
                    f.write(b'\n'); f.flush(); os.fsync(f.fileno())
        return count

    def append(self, row):
        row = {**row, **self.identity}
        key = self.key(row)
        assert key not in self.records
        if not self.chunks or self.active_rows == self.chunk_rows:
            self.chunks.append(self.directory / f'chunk-{len(self.chunks):06d}.jsonl')
            self.active_rows = 0
        # Same bytes and durability boundary as the frozen Journal.append.
        with self.chunks[-1].open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
            f.flush()
            os.fsync(f.fileno())
        self.records[key] = row
        self.active_rows += 1

    @property
    def path(self):
        """The unchanged lens runner asks for its canonical artifact only at completion."""
        tmp = self.canonical.with_suffix('.consolidating.tmp')
        expected = hashlib.sha256()
        with tmp.open('wb') as dest:
            for path in [self.prefix, *self.chunks]:
                with path.open('rb') as source:
                    for data in iter(lambda: source.read(1 << 20), b''):
                        expected.update(data); dest.write(data)
            dest.flush(); os.fsync(dest.fileno())
        assert sha(tmp) == expected.hexdigest()
        tmp.replace(self.canonical)
        atomic_json(self.directory / 'consolidated.json',
                    {'canonical_sha256': expected.hexdigest(), 'records': len(self.records),
                     'prefix_records_preserved': self.prefix_records, 'chunks': len(self.chunks),
                     'storage': json.loads(self.meta.read_text())})
        return self.canonical


if __name__ == '__main__':
    # Only the storage object used by the lens appendix is substituted. The
    # frozen runner files, Runtime, lens transport, ranks, loops and RNG stay intact.
    import gen27b_lenses
    gen27b_lenses.Journal = ChunkedJournal
    runpy.run_path(str(Path(__file__).with_name('gen27b_run.py')), run_name='__main__')
