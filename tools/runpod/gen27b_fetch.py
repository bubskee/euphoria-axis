"""Fetch pinned model weights to container disk and verify every selected file."""
import argparse
import fnmatch
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from gen27b_common import atomic_json, sha


def fetch(model, revision, directory):
    directory = Path(directory)
    info = HfApi().model_info(model, revision=revision, files_metadata=True)
    assert info.sha == revision
    patterns = ['*.safetensors', '*.json', '*.jinja', '*.model', '*.txt']
    siblings = [s for s in info.siblings if any(fnmatch.fnmatch(s.rfilename, p) for p in patterns)]
    snapshot_download(model, revision=revision, local_dir=str(directory), allow_patterns=patterns, max_workers=4)
    records = {}
    for item in siblings:
        path = directory / item.rfilename
        assert path.is_file() and path.stat().st_size == item.size, str(path)
        if item.lfs:
            assert sha(path) == item.lfs.sha256, str(path)
        else:
            content = path.read_bytes()
            digest = hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest()
            assert digest == item.blob_id, str(path)
        records[item.rfilename] = {'bytes': item.size, 'sha256': sha(path)}
    atomic_json(directory / 'VERIFIED.json', {'model_id': model, 'revision': revision, 'files': records})
    print('Verified', model, revision, len(records), 'files', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('model')
    p.add_argument('revision')
    p.add_argument('directory')
    a = p.parse_args()
    fetch(a.model, a.revision, a.directory)
